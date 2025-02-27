#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node
from rclpy.parameter import Parameter
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger
from coordinates.srv import GetXYZFromImage
from rclpy.time import Time

from snaak_manipulation.action import FollowTrajectory, Pickup, ManipulateIngredient, ReturnToHome
import time

class ExecuteIngredientManipulationServer(Node):

    def __init__(self):
        super().__init__('ingredient_manipulation_server')

        self.declare_parameter('ham_bin_id', 'bin1')
        self.declare_parameter('cheese_bin_id', 'bin2')
        self.declare_parameter('bread_bin_id', 'bin3')

        self.bin_id = {
            'cheese_bin_id': self.get_parameter('cheese_bin_id').value,
            'ham_bin_id': self.get_parameter('ham_bin_id').value,
            'bread_bin_id': self.get_parameter('bread_bin_id').value
        }

        self._manipulate_ingred_action_server = ActionServer(
            self,
            ManipulateIngredient,
            self.get_name() + '/manipulate_ingredient',
            self.execute_ingred_manipulation_callback
        )

        self._rth_action_server = ActionServer(
            self,
            ReturnToHome,
            self.get_name() +'/return_home',
            self.execute_rth_callback
        )

        self._traj_action_client = ActionClient(self, FollowTrajectory, self.get_name() + 'follow_trajectory')
        self._pickup_action_client = ActionClient(self, Pickup, self.get_name() + '/pickup')
        self._reset_arm_action_client = ActionClient(self, ReturnToHome, self.get_name() + '/reset_arm')
        self.wait_for_action_clients()

        self._disable_vacuum_client = self.create_client(Trigger, 'disable_vacuum')
        self._get_xyz_client = self.create_client(GetXYZFromImage, 'vision/get_pickup_point') #TODO confirm this
        self.wait_for_service_clients()

        # Adding parameter callback so we can select which ingredient is where on the fly (may be useful later on)
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info("Started Manipulation Primary Node")
        
        self.current_location = 'home'

        self.trajectory_map = {
            'home': {'bin1': 1, 'bin2': 2, 'bin3': 3, 'assembly': 4},
            'bin1': {'home': 5, 'assembly': 6},
            'bin2': {'home': 7, 'assembly': 8},
            'bin3': {'home': 9, 'assembly': 10},
            'assembly': {'home': 11, 'bin1': 12, 'bin2': 13, 'bin3': 14}
        }

        reset_success = self.reset_arm()
        if not reset_success:
            self.get_logger().info('Reset Arm Failed')
            rclpy.shutdown()

    def wait_for_service_clients(self):
        while not self._disable_vacuum_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Vacuum disable service not available, waiting...')
        self.get_logger().info('All services are ready!')


    def wait_for_action_clients(self):
        clients = [
            ('follow_trajectory', self._traj_action_client),
            ('pickup', self._pickup_action_client),
            ('reset_arm', self._reset_arm_action_client)
        ]
        
        for client_name, client in clients:
            self.get_logger().info(f'Waiting for {client_name} action client...')
            client.wait_for_server()
            self.get_logger().info(f'{client_name} action client is ready!')

        self.get_logger().info('All action clients are ready!')

    def parameters_callback(self, parameter_list):
        for parameter in parameter_list:
            self.bin_id[parameter.name] = parameter.value
            self.get_logger().info(f"Parameter '{parameter.name}' updated to: {parameter.value}")
        return rclpy.parameter.SetParametersResult(successful=True)
    
    def send_goal(self, action_client: ActionClient, action_goal):
        """Helper function to send a goal and handle the result"""
        action_client.wait_for_server()
        
        send_goal_future = action_client.send_goal_async(action_goal)
        
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected by the server')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            return True 
        else:
            return False    

    def execute_ingred_manipulation_callback(self, goal_handle):
        # Considering ingredient 0 is cheese, 1 is ham, and 2 is bread
        ingredient_id = goal_handle.request.ingredient_id
        bin_location = None
        self.get_logger().info(f"Manipulating {ingredient_id}...")

        if (ingredient_id == 'cheese'):
            bin_location = self.bin_id["cheese_bin_id"]
        elif (ingredient_id == 'ham'):
            bin_location = self.bin_id["ham_bin_id"]
        elif (ingredient_id == 'bread'):
            bin_location = self.bin_id["bread_bin_id"]

        if bin_location == None:
            self.get_logger().info("Invalid Ingredient ID Given")
            goal_handle.abort()
            return ManipulateIngredient()

        traj_id = self.trajectory_map[self.current_location][bin_location]
        traj_goal = FollowTrajectory.Goal()
        traj_goal.traj_id = traj_id

        traj_success = self.send_goal(self._traj_action_client, traj_goal)

        if (not traj_success):
            goal_handle.abort()
            self.get_logger().error("Trajectory Following Failed")
            return ManipulateIngredient.Result()
        else:
            self.current_location = bin_location

        time.sleep(2)


        # TODO Replace this with vision values, these have been manually found
        coordRequest = GetXYZFromImage.Request()
        coordRequest.bin_id = bin_location
        coordRequest.timestamp = Time()

        self.future = self._get_xyz_client.call_async(coordRequest)
        rclpy.spin_until_future_complete(self, self.future)
        result = self.future.result()

        if (result.x == -1):
            goal_handle.abort()
            self.get_logger().error("Vision Unable to Get XYZ")
            return ManipulateIngredient.Result()
        
        pickup_goal = Pickup.Goal()
        pickup_goal.x = result.x
        pickup_goal.y = result.y
        pickup_goal.z = result.depth
        self.get_logger().info(f"Camera XYZ: {pickup_goal.x}, {pickup_goal.y}, {pickup_goal.z}")
            
        pickup_success = self.send_goal(self._pickup_action_client, pickup_goal)
        result = self.future.result()
        if (not pickup_success):
            self.get_logger().error("Ingredient Pick-Up Failed")
            goal_handle.abort()
            return ManipulateIngredient.Result()

        # TODO Add weighing scale service call to determine if we actually grabbed the ingredient

        traj_id = self.trajectory_map[self.current_location]["assembly"]
        traj_goal = FollowTrajectory.Goal()
        traj_goal.traj_id = traj_id

        traj_success = self.send_goal(self._traj_action_client, traj_goal)

        if (not traj_success):
            goal_handle.abort()
            self.get_logger().error("Trajectory Following Failed")
            return ManipulateIngredient.Result()
        else:
            self.current_location = "assembly"

        time.sleep(2)
        disable_req = Trigger.Request()

        self.future = self._disable_vacuum_client.call_async(disable_req)
        rclpy.spin_until_future_complete(self, self.future)

        # TODO add place manuever
        goal_handle.succeed()
        self.get_logger().info("Manipulation Succesful!")


        return ManipulateIngredient.Result()

    def reset_arm(self):
        reset_goal = ReturnToHome.Goal()
        success = self.send_goal(self._reset_arm_action_client, reset_goal)
        return success
    
    def execute_rth_callback(self, goal_handle):
        result = ReturnToHome.Result()
        if self.current_location == 'home':
            success = self.reset_arm()
        else:
            traj_id = self.trajectory_map[self.current_location]['home']
            traj_goal = FollowTrajectory.Goal()
            traj_goal.traj_id = traj_id
            success = self.send_goal(self._traj_action_client, traj_goal)

        if (not success):
            goal_handle.abort()
            self.get_logger().error("Return To Home Failed")
            return result
        else:
            self.current_location = 'home'
            goal_handle.succeed()
            return result

def main(args=None):
    rclpy.init(args=args)
    ingredient_manipulation_server = ExecuteIngredientManipulationServer()

    try:
        rclpy.spin(ingredient_manipulation_server)
    except KeyboardInterrupt:
        pass
    finally:
        ingredient_manipulation_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()