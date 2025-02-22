#!/usr/bin/env python3
import numpy as np
import pickle, time
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from snaak_manipulation.action import FollowTrajectory, Pickup

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

        self.fa = FrankaArm(init_rclpy=False)
        self.get_logger().info("Started Manipulation Action Server Node")


    def execute_trajectory_callback(self, goal_handle):

        self.get_logger().info("Opening .pkl File...")
        traj_file_path = self.traj_id_to_file(goal_handle.request.traj_id)
        
        result = FollowTrajectory.Result()

        if traj_file_path is None or not self.fa.is_skill_done:
            goal_handle.abort()
            return result
        else:
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
            return result
    
    def wait_for_skill_with_collision_check(self):
        while(not self.fa.is_skill_done()):  # looping, and at each iteration detect if arm is in collision with boxes (this uses the frankapy boxes)
            if (self.fa.is_joints_in_collision_with_boxes()):
                self.fa.stop_skill() # this seems to make the motion break, but it does prevent collision
                raise Exception("In Collision with boxes, cancelling motion")

    def execute_pickup_callback(self, goal_handle):

        # first move to x, y
        # rotate to reset the end effector downwards
        # move down to pick up
        # send service/action call to start vaccuum and wait for response
        # move up

        success = False
        result = Pickup.Result()

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
            self.fa.goto_pose(new_pose, cartesian_impedances=[3000, 3000, 300, 300, 300, 300], use_impedance=False, block=False)
            self.get_logger().info("Moving above grasp point...")
            self.wait_for_skill_with_collision_check()
            
            # move down
            #new_pose = self.fa.get_pose()
            new_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
            new_pose.translation = [destination_x, destination_y, z_pre_grasp - depth] #x, y global, depth is relative TODO: confirm this
            new_pose.rotation = default_rotation
            self.fa.goto_pose(new_pose, cartesian_impedances=[3000, 3000, 300, 300, 300, 300], use_impedance=False, block=False)
            self.get_logger().info("Moving Down...")
            self.wait_for_skill_with_collision_check()

            # call the pneumatic node service
            # self.get_logger("Grasped!")
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
        match traj_id:
            case 1:
                pkl_file_name = "home2bin1_cam_verified.pkl"
            case 2:
                pkl_file_name = "home2bin2_cam_verified.pkl"
            case 3:
                pkl_file_name = "home2bin3_cam_verified.pkl"
            case 4:
                pkl_file_name = "home_assembly_traj.pkl"
        
        if pkl_file_name is None:
            self.get_logger().info('Invalid Trajectory Entered')
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


        # Goto the first position in the trajectory.
        #fa.log_info('Initializing Sensor Publisher')

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