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
from snaak_manipulation.action import FollowTrajectory, Pickup, ReturnToHome

from std_srvs.srv import Trigger
import os
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Transform, Vector3, Quaternion
import tf_transformations

from autolab_core import RigidTransform

class ManipulationActionServerNode(Node):
    def __init__(self):
        super().__init__('manipulation_action_server')
        self._traj_action_server = ActionServer(
            self,
            FollowTrajectory,
            'follow_trajectory',
            self.execute_trajectory_callback
        )

        self._pickup_action_server = ActionServer(
            self,
            Pickup,
            'pickup',
            self.execute_pickup_callback
        )

        self._reset_arm_action_server = ActionServer(
            self,
            ReturnToHome,
            'reset_arm',
            self.execute_reset_arm_callback
        )

        self._enable_vacuum_client = self.create_client(Trigger, 'enable_vacuum')
        while not self._enable_vacuum_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Vacuum enable service not available, waiting...')

        self._req = Trigger.Request()

        self.fa = FrankaArm(init_rclpy=False)

        self.trajectory_file_map = {
            1: 'home2bin1_verified.pkl',
            2: 'home2bin2_verified.pkl',
            3: 'home2bin3_verified.pkl',
            4: 'home2assembly_verified.pkl',
            5: 'bin12home_verified.pkl',
            6: 'bin12assembly_verified.pkl',
            7: 'bin22home_verified.pkl',
            8: 'bin22assembly_verified.pkl',
            9: 'bin32home_verified.pkl',
            10: 'bin32assembly_verified.pkl',
            11: 'assembly2home_verified.pkl',
            12: 'assembly2bin1_verified.pkl',
            13: 'assembly2bin2_verified.pkl',
            14: 'assembly2bin3_verified.pkl'
        }

        # See collision_boxes.txt in /franka_settings for more detailed explanation
        self.kiosk_collision_boxes = np.array([
            # sides
            [0.25, 0.55, 0.5, 0, 0, 0, 1.1, 0.01, 1.1],
            [0.25, -0.55, 0.5, 0, 0, 0, 1.1, 0.01, 1.1],
            # back
            [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1], 
            # front
            [0.77, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
            # top
            [0.25, 0, 1, 0, 0, 0, 1.2, 1, 0.01],
            # bottom
            [0.25, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01],
            
            # sandwich assembly area
            [0.5, 0.25, 0.125, 0, 0, 0, 0.68, 0.695, 0.26],
            
            # right bin area
            [0.43, -0.3615, 0.0, 0, 0, 0, 0.68, 0.375, 0.001],
            [0.14, -0.3615, 0.125, 0, 0, 0, 0.08, 0.375, 0.26],
            [0.344, -0.3615, 0.125, 0, 0, 0, 0.05, 0.375, 0.26],
            [0.542, -0.3615, 0.125, 0, 0, 0, 0.05, 0.375, 0.26],
            [0.75, -0.3615, 0.125, 0, 0, 0, 0.08, 0.375, 0.26],
            [0.43, -0.215, 0.125, 0, 0, 0, 0.68, 0.07, 0.26],
            [0.43, -0.52, 0.125, 0, 0, 0, 0.68, 0.07, 0.26]
        ])
        self.get_logger().info("Started Manipulation Action Server Node")


    def execute_trajectory_callback(self, goal_handle):

        traj_file_path = self.traj_id_to_file(goal_handle.request.traj_id)
        
        result = FollowTrajectory.Result()

        if traj_file_path is None:
            self.get_logger().error("Invalid Trajectory ID")
            goal_handle.abort()
            return result
        
        self.fa.wait_for_skill() # in case other skill is running

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
    
    def wait_for_skill_with_collision_check(self):
        while(not self.fa.is_skill_done()):  # looping, and at each iteration detect if arm is in collision with boxes (this uses the frankapy boxes)
            if (self.fa.is_joints_in_collision_with_boxes(boxes=self.kiosk_collision_boxes)):
                self.fa.stop_skill() # this seems to make the motion break, but it does prevent collision
                raise Exception("In Collision with boxes, cancelling motion")

    def execute_reset_arm_callback(self, goal_handle):
        self.get_logger().info('Resetting Arm')
        try:
            self.fa.reset_joints()
            goal_handle.succeed()
        except Exception as e:
            self.get_logger().info('Error During Return to Home')
            goal_handle.abort()
            raise e
        finally:
            return ReturnToHome.Result()


    def execute_pickup_callback(self, goal_handle):

        # first move to x, y
        # rotate to reset the end effector downwards
        # move down to pick up
        # send service/action call to start vaccuum and wait for response
        # move up
        success = False
        result = Pickup.Result()
        self.fa.wait_for_skill() # in case other skill is running
        try:
            destination_x = goal_handle.request.x
            destination_y = goal_handle.request.y
            depth = goal_handle.request.depth

            default_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            
            # move to x, y
            z_pre_grasp = self.fa.get_pose().translation[2]
            new_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
            new_pose.translation = [destination_x, destination_y, z_pre_grasp]
            new_pose.rotation = default_rotation
            self.fa.goto_pose(new_pose, use_impedance=False, block=False) # TODO Issue when going to furthest out bin
            self.get_logger().info("Moving above grasp point...")
            self.wait_for_skill_with_collision_check()
            
            # move down
            new_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
            new_pose.translation = [destination_x, destination_y, z_pre_grasp - depth] #x, y global, depth is relative
            new_pose.rotation = default_rotation
            self.fa.goto_pose(new_pose, cartesian_impedances=[3000, 3000, 300, 300, 300, 300], use_impedance=False, block=False)
            self.get_logger().info("Moving Down...")
            self.wait_for_skill_with_collision_check()

            # call the pneumatic node service
            # self.get_logger("Grasped!")
            self.future = self._enable_vacuum_client.call_async(self._req)
            rclpy.spin_until_future_complete(self, self.future)
            time.sleep(2)
            # move up
            new_pose = self.fa.get_pose()
            new_pose.translation[2] = z_pre_grasp
            new_pose.rotation = default_rotation
            self.fa.goto_pose(new_pose, cartesian_impedances=[3000, 3000, 300, 300, 300, 300], use_impedance=False, block=False)
            self.get_logger().info("Moving up...")
            self.wait_for_skill_with_collision_check()
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

    def traj_id_to_file(self, traj_id):
        package_share_directory = get_package_share_directory('snaak_manipulation')
        pkl_file_name = None
        if traj_id in self.trajectory_file_map:
            pkl_file_name = self.trajectory_file_map[traj_id]
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

        # go to initial pose if needed, this is more a safety feature, should not be relied on
        self.fa.goto_joints(joints_traj[0])
        self.fa.goto_joints(joints_traj[-1])
        # # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        # self.fa.goto_joints(joints_traj[1], duration=T, dynamic=True, buffer_time=10)
        # init_time = self.fa.get_time()
        # for i in range(2, len(joints_traj)):
        #     traj_gen_proto_msg = JointPositionSensorMessage(
        #         id=i, timestamp=self.fa.get_time() - init_time, 
        #         joints=joints_traj[i]
        #     )
        #     self.get_logger().info(f'joint angles: {joints_traj[i]}')

        #     ros_msg = make_sensor_group_msg(
        #         trajectory_generator_sensor_msg=sensor_proto2ros_msg(
        #             traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        #     )
            
        #     #fa.log_info('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        #     self.fa.publish_sensor_data(ros_msg)
        #     time.sleep(dt)
    
        # term_proto_msg = ShouldTerminateSensorMessage(timestamp=self.fa.get_time() - init_time, should_terminate=True)
        # ros_msg = make_sensor_group_msg(
        #     termination_handler_sensor_msg=sensor_proto2ros_msg(
        #         term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        #     )
        # self.fa.publish_sensor_data(ros_msg)
        # self.fa._in_skill = False
        # self.fa.stop_skill()


def main(args=None):
    rclpy.init(args=args)
    manipulation_action_server = ManipulationActionServerNode()
    try:
        rclpy.spin(manipulation_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        manipulation_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()