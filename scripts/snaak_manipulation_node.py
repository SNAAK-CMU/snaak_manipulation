#!/usr/bin/env python3
import numpy as np
import pickle, time
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage, PosePositionSensorMessage
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from snaak_manipulation.action import ExecuteTrajectory, Pickup, ReturnHome, Place

from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Transform, Vector3, Quaternion
import tf_transformations
from autolab_core import RigidTransform
from example_interfaces.srv import SetBool
import asyncio
from scripts.snaak_manipulation_utils import pickup_traj, get_traj_file
import sys
from tf2_msgs.msg import TFMessage
import copy



from scripts.snaak_manipulation_constants import KIOSK_COLLISION_BOXES

class ManipulationActionServerNode(Node):
    def __init__(self):
        super().__init__('snaak_manipulation')

        # TODO transfer these into FSM 
        self.declare_parameter('ham_bin_id', 'bin1')
        self.declare_parameter('cheese_bin_id', 'bin2')
        self.declare_parameter('bread_bin_id', 'bin3')
        self.declare_parameter('assembly_tray_id', '4')
        self.declare_parameter('assembly_bread_id', '5')

        self.location_id = {
            'cheese_bin_id': self.get_parameter('cheese_bin_id').value,
            'ham_bin_id': self.get_parameter('ham_bin_id').value,
            'bread_bin_id': self.get_parameter('bread_bin_id').value,
            'assembly_tray_id': self.get_parameter('assembly_tray_id').value,
            'assembly_bread_id': self.get_parameter('assembly_bread_id').value
        }
        self.add_on_set_parameters_callback(self.parameters_callback)

        self._traj_action_server = ActionServer(
            self,
            ExecuteTrajectory,
            'snaak_manipulation/execute_trajectory',
            self.execute_trajectory_callback
        )

        self._pickup_action_server = ActionServer(
            self,
            Pickup,
            'snaak_manipulation/pickup',
            self.execute_pickup_callback
        )

        self._place_action_server = ActionServer(
            self,
            Place,
            'snaak_manipulation/place',
            self.execute_place_callback
        )

        self._rth_action_server = ActionServer(
            self,
            ReturnHome,
            'snaak_manipulation/return_home',
            self.execute_rth_callback
        )

        self.subscription_tf = self.create_subscription(
            TFMessage, "/tf", self.tf_listener_callback_tf, 10
        )

        self._enable_srv = self.create_service(Trigger, 'snaak_manipulation/enable_arm', self.enable_callback)
        self._disable_srv = self.create_service(Trigger, 'snaak_manipulation/disable_arm', self.disable_callback)

        self._disable_vacuum_client = self.create_client(Trigger, '/snaak_pneumatic/disable_vacuum')
        self._enable_vacuum_client = self.create_client(Trigger, '/snaak_pneumatic/enable_vacuum')
        self._eject_vacuum_client = self.create_client(SetBool, '/snaak_pneumatic/eject_vacuum')

        self.wait_for_service_clients()

        self.fa = FrankaArm(init_rclpy=False)
        self.pre_grasp_height = 0.3
        self.pickup_place_impedances = [2000.0, 2000.0, 600.0, 50.0, 50.0, 50.0]

        self.collision_detected = False
        self.current_location = 'home'
        self.arm_enabled = True
        self.prev_tf = None
        self.transformations = {}

    def tf_listener_callback_tf(self, msg):
        """Handle incoming transform messages."""
        for transform in msg.transforms:
            if transform.child_frame_id and transform.header.frame_id:
                self.transformations[
                    (transform.header.frame_id, transform.child_frame_id)
                ] = transform

    def wait_for_service_clients(self):
        clients = [
            ('disable_vaccuum', self._disable_vacuum_client),
            ('enable_vaccuum', self._enable_vacuum_client),
            ('eject_vaccum', self._eject_vacuum_client)
        ]
        
        for client_name, client in clients:
            self.get_logger().info(f'Waiting for {client_name} action client...')
            client.wait_for_service()
            self.get_logger().info(f'{client_name} action client is ready!')

        self.get_logger().info('All service clients are ready!')

    # TODO transfer this into FSM
    def parameters_callback(self, parameter_list):
        for parameter in parameter_list:
            self.location_id_id[parameter.name] = parameter.value
            self.get_logger().info(f"Parameter '{parameter.name}' updated to: {parameter.value}")
        return rclpy.parameter.SetParametersResult(successful=True)
    
    def wait_for_skill_with_collision_check(self, validate=True):
        #start_time = time.time()
        while(not self.fa.is_skill_done()):
            if (self.fa.is_joints_in_collision_with_boxes(boxes=KIOSK_COLLISION_BOXES)):
                self.fa.stop_skill()
                self.fa.wait_for_skill()
                raise Exception("In Collision with boxes, cancelling motion...")
            # if (time.time() - start_time > 10):
            #     self.fa.stop_skill()
            #     self.fa.wait_for_skill()
            #     raise Exception("Timed out, arm not responsive...")
            time.sleep(0.01)
        #if validate: self.validate_execution()
            
    # def validate_execution(self, desired_pose=None, desired_joints=None, use_joints=False):
    #     """Raise exception if not reaching desired position"""
    #     if use_joints:
    #         curr_joints = self.fa.get_joints()
    #         if np.linalg.norm(desired_joints - curr_joints) > 0.25: # TODO: tune these parameters
    #             raise Exception("Did not reach desired joints")
    #     else:
    #         curr_translation = self.fa.get_pose().translation
    #         desired_translation = desired_pose.translation
    #         if np.linalg.norm(desired_translation - curr_translation) > 0.15:
    #             raise Exception("Did not reach desired pose")
            
    def validate_execution(self):
        """Raise exception if arm not responding"""
        transform_name = ("panda_link0", "panda_hand")
        curr_tf = copy.deepcopy(
                    self.transformations[transform_name].transform
                )
        curr_translation = np.array([
            curr_tf.translation.x,
            curr_tf.translation.y,
            curr_tf.translation.z,
        ])
        prev_translation = np.array([
            self.prev_tf.translation.x,
            self.prev_tf.translation.y,
            self.prev_tf.translation.z
        ])
        if (np.linalg.norm(prev_translation - curr_translation) < 0.01):
            raise Exception("Arm not responsive...")

    def get_prev_tf(self):
        transform_name = ("panda_link0", "panda_hand")
        self.prev_tf = copy.deepcopy(
            self.transformations[transform_name].transform
        )
        time.sleep(0.5)

    async def async_collision_check(self, boxes, dt):
        """Asynchronous collision check"""
        while not self.collision_detected:
            if self.fa.is_joints_in_collision_with_boxes(boxes=boxes):
                self.get_logger().error(f"In collision with boxes, stopping...")
                self.collision_detected = True
                return
            await asyncio.sleep(dt)

    def enable_callback(self, request, response):
        self.arm_enabled = True
        response.success = True
        response.message = "Arm enabled"
        return response
    
    def disable_callback(self, request, response):
        self.arm_enabled = False
        response.success = True
        response.message = "Arm disabled"
        return response

    def execute_joint_trajectory(self, traj_file_path):
        with open(traj_file_path, 'rb') as pkl_f:
            skill_data = pickle.load(pkl_f)
        
        assert skill_data[0]['skill_description'] == 'GuideMode', \
            "Trajectory not collected in guide mode"
        skill_state_dict = skill_data[0]['skill_state_dict']

        dt = 0.01

        joints_traj = skill_state_dict['q']
        T = len(joints_traj) * dt

        self.fa.wait_for_skill()
        self.collision_detected = False

        # go to initial pose if needed, this is more a safety feature, should not be relied on
        curr_joints = self.fa.get_joints()
        if (np.linalg.norm(curr_joints - joints_traj[0]) > 0.04):
            self.get_logger().info("Moving to start of trajectory...")
            #self.get_prev_tf()
            self.fa.goto_joints(joints_traj[0], use_impedance=False, block=False)
            self.wait_for_skill_with_collision_check(validate=False)

        collision_task = asyncio.run_coroutine_threadsafe(
            self.async_collision_check(KIOSK_COLLISION_BOXES, dt), asyncio.get_event_loop()
        )  
        #self.get_prev_tf()
        self.fa.goto_joints(joints_traj[1], duration=T, dynamic=True, buffer_time=1)
        init_time = self.fa.get_time()
        for i in range(2, len(joints_traj)):
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=i, timestamp=self.fa.get_time() - init_time, 
                joints=joints_traj[i]
            )
            
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            if (self.collision_detected):
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
        collision_task.cancel()
        if self.collision_detected:
            self.collision_detected = False
            raise Exception("In Collision with boxes, cancelling motion")
        
        #self.validate_execution()


    def execute_trajectory_callback(self, goal_handle):
        if not self.arm_enabled:
            goal_handle.abort()
            self.get_logger().error("Arm Disabled")
            return ExecuteTrajectory.Result()
        
        share_directory = get_package_share_directory('snaak_manipulation')
        desired_end_location = goal_handle.request.desired_location
        result = ExecuteTrajectory.Result()
        success = False
        try:
            if self.current_location != desired_end_location:
                traj_file_path = get_traj_file(share_directory, self.current_location, desired_end_location)
                
                if traj_file_path is None:
                    self.get_logger().error("Invalid Trajectory")
                    goal_handle.abort()
                    return result
                self.get_logger().info('Executing Trajectory...')
                self.execute_joint_trajectory(traj_file_path)
                self.current_location = desired_end_location
                success = True
            else:
                self.get_logger().info("Already at desired location")
                success = True
        except Exception as e:
            self.get_logger().error(f"Error Occured during trajectory following {e} ")
            goal_handle.abort()
            raise e
        finally:
            if success:
                goal_handle.succeed()
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
        

    def execute_pose_trajectory(self, pose_traj, dt, T, at_start=True):
        '''
        Follow a pose trajectory based on a list of rigid transforms

        CAUTION: YOU MUST BE AT START X, Y, Z TO SAFELY CALL THIS FUNCTION\n
        If not, set at_start flag to false

        Inputs:
            pose_traj: list of rigid transforms
            dt: time between publishing
            T: time duration of pose trajectory

        Outputs:
            none
        '''
        if not at_start:
            #self.get_prev_tf()
            self.fa.goto_pose(pose_traj[0], 
                        duration=4.0, 
                        use_impedance=False,
                        block=False,
                        cartesian_impedances=self.pickup_place_impedances)
            self.wait_for_skill_with_collision_check()

        #self.get_prev_tf()
        self.fa.goto_pose(pose_traj[1], 
                    duration=T, 
                    dynamic=True, 
                    buffer_time=1, 
                    use_impedance=False,
                    cartesian_impedances=[2000.0, 2000.0, 600.0, 50.0, 50.0, 50.0]
        )
        self.collision_detected = False

        # execute collision in sseperate thread
        collision_task = asyncio.run_coroutine_threadsafe(
            self.async_collision_check(KIOSK_COLLISION_BOXES, dt), asyncio.get_event_loop()
        )        
        init_time = self.fa.get_time()
        for i in range(2, len(pose_traj)):
            timestamp = self.fa.get_time() - init_time
            pose_tf = pose_traj[i]
            traj_gen_proto_msg = PosePositionSensorMessage(
                id=i, 
                timestamp=timestamp,
                position=pose_tf.translation, 
                quaternion=pose_tf.quaternion
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, 
                    SensorDataMessageType.POSE_POSITION),
                )
            if self.collision_detected:
                break
            self.fa.publish_sensor_data(ros_msg)
            time.sleep(dt)

        term_proto_msg = ShouldTerminateSensorMessage(timestamp=self.fa.get_time() - init_time, 
                                                    should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        
        self.fa.publish_sensor_data(ros_msg)
        self.fa.wait_for_skill()
        collision_task.cancel()
        if self.collision_detected:
            self.collision_detected = False
            raise Exception("In Collision with boxes, cancelling motion")
        #self.validate_execution()

    def execute_pickup(self, pickup_point):
        '''
        Executes pickup sequence

        Inputs:
            pickup_point: goal pickup point
        
        Outpus:
            none
        '''
        self.fa.wait_for_skill() 
        pre_grasp_joints = self.fa.get_joints()
        destination_x, destination_y, destination_z = pickup_point
        # TODO put z offset here?
        default_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        # move to x, y, and z directly above the bin
        new_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
        new_pose.translation = [destination_x, destination_y, self.pre_grasp_height]
        new_pose.rotation = default_rotation
        self.fa.goto_pose(new_pose, cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, use_impedance=False, block=False)
        self.get_logger().info("Moving above grasp point...")
        #self.get_prev_tf()
        self.wait_for_skill_with_collision_check()

        # move down
        self.get_logger().info("Moving Down...")
        curr_z = self.fa.get_pose().translation[2]
        pose_traj, dt, T = pickup_traj(destination_x, destination_y, curr_z, destination_z)
        self.execute_pose_trajectory(pose_traj, dt, T)
        enable_req = Trigger.Request()

        self.future = self._enable_vacuum_client.call_async(enable_req)
        rclpy.spin_until_future_complete(self, self.future)
        time.sleep(2)

        # move up
        self.get_logger().info("Moving up...")
        curr_z = self.fa.get_pose().translation[2]
        pose_traj, dt, T = pickup_traj(destination_x, destination_y, curr_z, self.pre_grasp_height)
        self.execute_pose_trajectory(pose_traj, dt, T)

        # move to pre-grasp pose
        self.fa.goto_joints(pre_grasp_joints, use_impedance=False, block=False)
        #self.get_prev_tf()
        self.wait_for_skill_with_collision_check()

    def execute_pickup_callback(self, goal_handle):
        success = False
        result = Pickup.Result()
        if not self.arm_enabled:
            goal_handle.abort()
            self.get_logger().error("Arm Disabled")
            return result
        
        ingredient_type = goal_handle.request.ingredient_type # TODO: Integrate this if need seperate pickup techniques

        try:
            destination_x = goal_handle.request.x
            destination_y = goal_handle.request.y
            destination_z = goal_handle.request.z
            self.execute_pickup((destination_x, destination_y, destination_z))
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

    def execute_place_callback(self, goal_handle):
        success = False
        result = Place.Result()
        if not self.arm_enabled:
            goal_handle.abort()
            self.get_logger().error("Arm Disabled")
            return result
        try:
            destination_x = goal_handle.request.x
            destination_y = goal_handle.request.y
            destination_z = goal_handle.request.z
            ingredient_type = goal_handle.request.ingredient_type
            if ingredient_type == 1:
                self.execute_place_sliced((destination_x, destination_y, destination_z))
                success=True
            elif ingredient_type == 2:
                #TODO call function for bread placement maneuver
                success = False
            elif ingredient_type == 3:
                #TODO call function for shredded ingredient placement maneuver
                success = False
            else:
                raise "Invalid Ingredient Type"
        except Exception as e:
            self.get_logger().error(f"Error Occured during place motion {e} ")
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
        
    def execute_place_sliced(self, place_point):
        '''
        Execute place sequence

        Inputs:
            place_point: desired place point of ingredient
        
        Outputs:
            none
        '''

        self.fa.wait_for_skill()
        check_joints = self.fa.get_joints()
        self.get_logger().info("Executing Sliced Ingredient Place maneuver...")
        destination_x, destination_y, destination_z = place_point
        default_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # move to x, y, (z + 0.05)
        new_pose = RigidTransform(from_frame='franka_tool', to_frame='world')
        new_pose.translation = [destination_x, destination_y, destination_z+0.05]
        new_pose.rotation = default_rotation
        self.fa.goto_pose(new_pose, cartesian_impedances=self.pickup_place_impedances, use_impedance=False, block=False) # TODO Change impedances?
        self.get_logger().info("Moving above release point...")
        #self.get_prev_tf()
        self.wait_for_skill_with_collision_check()

        
        # disable vacuum
        disable_req = Trigger.Request()
        self.future = self._disable_vacuum_client.call_async(disable_req)
        rclpy.spin_until_future_complete(self, self.future)

        # release ingredient
        eject_req = SetBool.Request()
        eject_req.data = True
        self.future = self._eject_vacuum_client.call_async(eject_req)
        rclpy.spin_until_future_complete(self, self.future)

        #TODO add go to pre-place position and execute collision check
        self.get_logger().info("Moving back to check position...")
        self.fa.goto_joints(check_joints, cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, use_impedance=False, block=False)
        #self.get_prev_tf()
        self.wait_for_skill_with_collision_check()


    def reset_arm(self):
        try:
            #self.get_prev_tf()
            self.fa.reset_joints(block=False)
            self.wait_for_skill_with_collision_check()
            #self.validate_execution()
        except:
            return False
        finally:
            self.current_location = "home"
            return True

    def execute_rth_callback(self, goal_handle):
        if not self.arm_enabled:
            goal_handle.abort()
            self.get_logger().error("Arm Disabled")
            return ReturnHome.Result()

        success = True
        if self.current_location == 'home':
            success = self.reset_arm()
        else:
            share_directory = get_package_share_directory('snaak_manipulation')
            traj_file_path = get_traj_file(share_directory, self.current_location, 'home')

            try:
                self.execute_joint_trajectory(traj_file_path)
            except:
                success = False

        if (not success):
            goal_handle.abort()
            self.get_logger().error("Return To Home Failed")
            return ReturnHome.Result()
        else:
            self.current_location = 'home'
            goal_handle.succeed()
            return ReturnHome.Result()
            
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
