#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node
from rclpy.parameter import Parameter
from action_msgs.msg import GoalStatus

from snaak_manipulation.action import FollowTrajectory, Pickup, ManipulateIngredient, ReturnToHome
import time

class ExecuteIngredientManipulationServer(Node):

    def __init__(self):
        super().__init__('ingredient_manipulation_server')

        self.declare_parameter('ham_bin_id', 1)
        self.declare_parameter('cheese_bin_id', 2)
        self.declare_parameter('bread_bin_id', 3)

        self.bin_id = {
            'cheese_bin_id': self.get_parameter('cheese_bin_id').value,
            'ham_bin_id': self.get_parameter('ham_bin_id').value,
            'bread_bin_id': self.get_parameter('bread_bin_id').value
        }

        self._action_server = ActionServer(
            self,
            ManipulateIngredient,
            'manipulate_ingredient',
            self.execute_ingred_manipulation_callback
        )

        self._action_server = ActionServer(
            self,
            ReturnToHome,
            'return_home',
            self.execute_rth_callback
        )

        self._traj_action_client = ActionClient(self, FollowTrajectory, 'follow_trajectory')
        self._pickup_action_client = ActionClient(self, Pickup, 'pickup')
        
        # Adding parameter callback so we can select which ingredient is where on the fly (may be useful later on)
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info("Started Manipulation Primary Node")
        
        self.current_location = 0  # 0 is home, 1 - 3 are bins, 4 is assembly

        self.trajectory_map = {
            0: {1: 1, 2: 2, 3: 3, 4: 4},    # From home to bins 1-3 and assembly
            1: {0: 5, 4: 6},                # From bin 1 to home and assembly
            2: {0: 7, 4: 8},                # From bin 2 to home and assembly
            3: {0: 9, 4: 10},               # From bin 3 to home and assembly
            4: {0: 11, 1: 12, 2: 13, 3: 14} # From assembly to bins 1-3 and home
        }

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
        self.current_location = 0 # TODO this is temporary, until we have more trajectories generated
        bin_location = None
        match ingredient_id:
            case 0:
                bin_location = self.bin_id["cheese_bin_id"]
            case 1:
                bin_location = self.bin_id["ham_bin_id"]
            case 2:
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

        pickup_goal = Pickup.Goal()

        # TODO Replace this with vision values, these have been manually found
        if (bin_location == 2):
            pickup_goal.x = 0.447 
            pickup_goal.y = -0.302
            pickup_goal.depth = 0.23
        elif (bin_location == 3):
            pickup_goal.x = 0.24
            pickup_goal.y = -0.302
            pickup_goal.depth = 0.3

        pickup_success = self.send_goal(self._pickup_action_client, pickup_goal)

        if (not pickup_success):
            self.get_logger().error("Ingredient Pick-Up Failed")
            goal_handle.abort()
            return ManipulateIngredient.Result()
        else:
            self.current_location = bin_location

        goal_handle.succeed()


        # TODO add trajectory following to assembly area and placing ingredient
        return ManipulateIngredient.Result()


    def execute_rth_callback(self, goal_handle):
        traj_id = self.trajectory_map[self.current_location][0]
        traj_goal = FollowTrajectory.Goal()
        traj_goal.traj_id = traj_id


        success = self.send_goal(self._traj_action_client, traj_goal)
        if (not success):
            goal_handle.abort()
            self.get_logger().error("Return To Home Failed")
            return ReturnToHome.Result()
        else:
            self.current_location = 0

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