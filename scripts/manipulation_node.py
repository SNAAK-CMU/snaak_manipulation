#!/usr/bin/env python3
import numpy as np
import pickle, time
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from snaak_manipulation.action import FollowTrajectory, Pickup, ReturnToHome, ManipulateIngredient
from snaak_vision.srv import GetXYZFromImage

from std_srvs.srv import Trigger
import os
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Transform, Vector3, Quaternion
import tf_transformations
from autolab_core import RigidTransform
import sys

from scripts.manipulation_constants import TRAJECTORY_FILE_MAP, TRAJECTORY_MAP, KIOSK_COLLISION_BOXES

class ManipulationActionServerNode(Node):
    def __init__(self):
        super().__init__('manipulation_action_server')


        self.declare_parameter('ham_bin_id', 'bin1')
        self.declare_parameter('cheese_bin_id', 'bin2')
        self.declare_parameter('bread_bin_id', 'bin3')

        self.bin_id = {
            'cheese_bin_id': self.get_parameter('cheese_bin_id').value,
            'ham_bin_id': self.get_parameter('ham_bin_id').value,
            'bread_bin_id': self.get_parameter('bread_bin_id').value
        }
        self.add_on_set_parameters_callback(self.parameters_callback)

        self._traj_action_server = ActionServer(
            self,
            FollowTrajectory,
            'snaak_manipulation/follow_trajectory',
            self.execute_trajectory_callback
        )

        self._pickup_action_server = ActionServer(
            self,
            Pickup,
            'snaak_manipulation/pickup',
            self.execute_pickup_callback
        )

        self._manipulate_ingred_action_server = ActionServer(
            self,
            ManipulateIngredient,
            'snaak_manipulation/manipulate_ingredient',
            self.execute_ingred_manipulation_callback
        )

        self._rth_action_server = ActionServer(
            self,
            ReturnToHome,
            'snaak_manipulation/return_home',
            self.execute_rth_callback
        )
        
        self._disable_vacuum_client = self.create_client(Trigger, 'disable_vacuum')
        self._enable_vacuum_client = self.create_client(Trigger, 'enable_vacuum')

        self._get_xyz_client = self.create_client(GetXYZFromImage, 'vision_node/get_pickup_point')
        self.wait_for_service_clients()

        self.get_logger().info("Started Manipulation Node")
        

        self.fa = FrankaArm(init_rclpy=False)
        self.pre_grasp_height = 0.3

        # TODO put constants in a different location
        self.current_location = 'home'

        reset_success = self.reset_arm()
        if not reset_success:
            self.get_logger().info('Reset Arm Failed')
            rclpy.shutdown()

    def wait_for_service_clients(self):
        clients = [
            ('disable_vaccuum', self._disable_vacuum_client),
            ('enable_vaccuum', self._enable_vacuum_client),
            ('vision_node/get_pickup_point', self._get_xyz_client)
        ]
        
        for client_name, client in clients:
            self.get_logger().info(f'Waiting for {client_name} action client...')
            client.wait_for_service()
            self.get_logger().info(f'{client_name} action client is ready!')

        self.get_logger().info('All service clients are ready!')

    
    def parameters_callback(self, parameter_list):
        for parameter in parameter_list:
            self.bin_id[parameter.name] = parameter.value
            self.get_logger().info(f"Parameter '{parameter.name}' updated to: {parameter.value}")
        return rclpy.parameter.SetParametersResult(successful=True)
    
    def wait_for_skill_with_collision_check(self):
        while(not self.fa.is_skill_done()):  # looping, and at each iteration detect if arm is in collision with boxes (this uses the frankapy boxes)
            if (self.fa.is_joints_in_collision_with_boxes(boxes=KIOSK_COLLISION_BOXES)):
                self.fa.stop_skill() # this seems to make the motion break, but it does prevent collision
                self.fa.wait_for_skill()
                raise Exception("In Collision with boxes, cancelling motion")
            
    def reset_arm(self):
        try:
            self.fa.reset_joints()
        except:
            return False
        finally:
            self.current_location = "home"
            return True
        
    def get_pickup_point(self, bin_location):
        coordRequest = GetXYZFromImage.Request()
        coordRequest.bin_id = int(bin_location[-1])
        coordRequest.timestamp = 1.0

        self.future = self._get_xyz_client.call_async(coordRequest)
        rclpy.spin_until_future_complete(self, self.future)
        result = self.future.result()

        if (result.x == -1):
            self.get_logger().error("Unable to Get XYZ from Vision Node")
            return None

        self.get_logger().info(f"Result from Vision Node: {result.x}, {result.y}, {result.z}")

        return result
    
    def traj_id_to_file(self, traj_id):
        package_share_directory = get_package_share_directory('snaak_manipulation')
        pkl_file_name = None
        if traj_id in TRAJECTORY_FILE_MAP:
            pkl_file_name = TRAJECTORY_FILE_MAP[traj_id]
        else:
            return None
        
        traj_file_path = os.path.join(package_share_directory, pkl_file_name)
        return traj_file_path

    def execute_trajectory(self, traj_file_path):
        with open(traj_file_path, 'rb') as pkl_f:
            skill_data = pickle.load(pkl_f)
        
        assert skill_data[0]['skill_description'] == 'GuideMode', \
            "Trajectory not collected in guide mode"
        skill_state_dict = skill_data[0]['skill_state_dict']

        T = float(skill_state_dict['time_since_skill_started'][-1])
        dt = 0.01

        joints_traj = skill_state_dict['q']

        self.fa.wait_for_skill()
        # go to initial pose if needed, this is more a safety feature, should not be relied on
        self.fa.goto_joints(joints_traj[0])
        self.fa.goto_joints(joints_traj[1], duration=T, dynamic=True, buffer_time=1)
        init_time = self.fa.get_time()
        success = True
        for i in range(2, len(joints_traj)):
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=i, timestamp=self.fa.get_time() - init_time, 
                joints=joints_traj[i]
            )
            
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )

            if (self.fa.is_joints_in_collision_with_boxes(boxes=KIOSK_COLLISION_BOXES)):
                success = False
                break
            
            self.fa.publish_sensor_data(ros_msg)
            time.sleep(dt)
    
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=self.fa.get_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.fa.publish_sensor_data(ros_msg)
        self.fa.wait_for_skill()

        if not success:
            raise Exception("In Collision with boxes, cancelling motion")


    def execute_trajectory_callback(self, goal_handle):
        traj_file_path = self.traj_id_to_file(goal_handle.request.traj_id)
        result = FollowTrajectory.Result()
        if traj_file_path is None:
            self.get_logger().error("Invalid Trajectory ID")
            goal_handle.abort()
            return result
        try:
            self.get_logger().info('Executing Trajectory...')
            self.execute_trajectory(traj_file_path)

            pose = self.fa.get_pose()
            transform = Transform()
            transform.translation = Vector3(
                x=pose.translation[0],
                y=pose.translation[1],
                z=pose.translation[2]
            )

            rotation_matrix = pose.rotation
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix

            q = tf_transformations.quaternion_from_matrix(transformation_matrix)
            transform.rotation = Quaternion(
                x=q[0],
                y=q[1],
                z=q[2],
                w=q[3]
            )

            goal_handle.succeed()
            result.end_pose = transform
        except Exception as e:
            self.get_logger().error(f"Error Occured during trajectory following {e} ")
            goal_handle.abort()
            raise e
        finally:
            return result

    def execute_pickup(self, pickup_point):
        # first move to x, y
        # rotate to reset the end effector downwards
        # move down to pick up
        # send service/action call to start vaccuum and wait for response
        # move up
        self.fa.wait_for_skill() 
        destination_x, destination_y, destination_z = pickup_point
        default_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        # move to x, y, and z directly above the bin
        new_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
        new_pose.translation = [destination_x, destination_y, self.pre_grasp_height]
        new_pose.rotation = default_rotation
        self.fa.goto_pose(new_pose, cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, use_impedance=False, block=False) # TODO Issue when going to furthest out bin
        self.get_logger().info("Moving above grasp point...")
        self.wait_for_skill_with_collision_check()

        # move down
        new_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
        new_pose.translation = [destination_x, destination_y, destination_z] #x, y global, depth is relative to grasp height
        new_pose.rotation = default_rotation
        self.fa.goto_pose(new_pose, cartesian_impedances=[3000, 3000, 300, 300, 300, 300], use_impedance=False, block=False)
        self.get_logger().info("Moving Down...")
        self.wait_for_skill_with_collision_check()

        enable_req = Trigger.Request()

        self.future = self._enable_vacuum_client.call_async(enable_req)
        rclpy.spin_until_future_complete(self, self.future)
        time.sleep(2)

        new_pose = self.fa.get_pose()
        new_pose.translation[2] = self.pre_grasp_height
        new_pose.rotation = default_rotation
        self.fa.goto_pose(new_pose, cartesian_impedances=[3000, 3000, 300, 300, 300, 300], use_impedance=False, block=False)
        self.get_logger().info("Moving up...")
        self.wait_for_skill_with_collision_check()
    
    def execute_pickup_callback(self, goal_handle):
        success = False
        result = Pickup.Result()
        try:
            destination_x = goal_handle.request.x
            destination_y = goal_handle.request.y
            destination_z = goal_handle.request.z
            self.execute_pickup(self, (destination_x, destination_y, destination_z))
            success=True
        except Exception as e:
            self.get_logger().error(f"Error Occured during pickup motion {e} ")
            goal_handle.abort()
            raise e
        finally:
            if success: goal_handle.succeed()
            pose = self.fa.get_pose()
            transform = Transform()
            transform.translation = Vector3(
                x=pose.translation[0],
                y=pose.translation[1],
                z=pose.translation[2]
            )

            rotation_matrix = pose.rotation
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix

            q = tf_transformations.quaternion_from_matrix(transformation_matrix)
            transform.rotation = Quaternion(
                x=q[0],
                y=q[1],
                z=q[2],
                w=q[3]
            )
            result.end_pose = transform

            return result  
        
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
        
        if not self.current_location == bin_location:
            traj_id = TRAJECTORY_MAP[self.current_location][bin_location]
            traj_file_path = self.traj_id_to_file(traj_id)
            try:
                self.execute_trajectory(traj_file_path)
            except:
                goal_handle.abort()
                self.get_logger().error("Trajectory Following Failed")
                return ManipulateIngredient.Result()
            finally:
                self.current_location = bin_location
                time.sleep(2)
        
        pickup_point = self.get_pickup_point(bin_location)
        if pickup_point == None:
            self.get_logger().error("Could Not Get Pickup Point")
            goal_handle.abort()
            return ManipulateIngredient.Result()
            
        try:
            self.execute_pickup((pickup_point.x, pickup_point.y, pickup_point.z))
        except:
            self.get_logger().error("Ingredient Pick-Up Failed")
            goal_handle.abort()
            return ManipulateIngredient.Result()

        # TODO Add weighing scale service call to determine if we actually grabbed the ingredient

        traj_id = TRAJECTORY_MAP[self.current_location]["assembly"]
        traj_file_path = self.traj_id_to_file(traj_id)
        try:
            self.execute_trajectory(traj_file_path)
        except:
            goal_handle.abort()
            self.get_logger().error("Trajectory Following Failed")
            return ManipulateIngredient.Result()
        finally:
            self.current_location = "assembly"
            time.sleep(2)

        disable_req = Trigger.Request() # TODO put this into place manuever
        self.future = self._disable_vacuum_client.call_async(disable_req)
        rclpy.spin_until_future_complete(self, self.future)

        # TODO add place manuever
        goal_handle.succeed()
        self.get_logger().info("Manipulation Succesful!")

        return ManipulateIngredient.Result()
    
    def execute_rth_callback(self, goal_handle):
        success = True
        if self.current_location == 'home':
            success = self.reset_arm()
        else:
            traj_id = TRAJECTORY_MAP[self.current_location]['home']
            traj_file_path = self.traj_id_to_file(traj_id)

            try:
                self.execute_trajectory(traj_file_path)
            except:
                success = False

        if (not success):
            goal_handle.abort()
            self.get_logger().error("Return To Home Failed")
            return ReturnToHome.Result()
        else:
            self.current_location = 'home'
            goal_handle.succeed()
            return ReturnToHome.Result()
        
def main(args=None):
    # TODO add proper shutdown with FrankaPy
    rclpy.init(args=args)
    manipulation_action_server = ManipulationActionServerNode()
    try:
        rclpy.spin(manipulation_action_server)
    except Exception as e:
        manipulation_action_server.get_logger().error(f'Error occurred: {e}')
    except KeyboardInterrupt:
        manipulation_action_server.get_logger().info('Keyboard interrupt received, shutting down...')
    finally:
        manipulation_action_server.fa.stop_robot_immediately()
        manipulation_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping Gracefully...")
        sys.exit(0)