import numpy as np
from scipy.integrate import cumtrapz
from autolab_core import RigidTransform
from scripts.snaak_manipulation_constants import TRAJECTORY_FILE_MAP, TRAJECTORY_ID_MAP
import os
import pickle
import yaml
from ament_index_python.packages import get_package_share_directory


def pickup_traj(x, y, start_z, end_z, step_size=0.001, acceleration = 0.1):
    '''
    Generates a trajectory from the current x, y, start_z, to x, y, end_z 
    using a trapazoidal velocity profile.

    Inputs:
        x: desired x position
        y: desired y position
        end_z: desired end z position in franka base link frame
        step_size: maximum z displacement that occur in one time step (0.01 s)
        acceleration: maximum allowable acceleration
    
    Outputs:
        pose_traj: calculated pose
        T: length of pose trajectory
        dt: delay between each pose
    '''

    if abs(start_z - end_z) < step_size:
        return
    
    default_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    total_distance = abs(end_z - start_z)
    direction = 1 if end_z > start_z else -1

    dt = 0.01
    max_velocity = step_size / dt

    t_accel = max_velocity / acceleration # time for robot to get up to speed
    d_accel = 0.5 * acceleration * t_accel**2 # distance to get up to max speed or return from max speed to 0
    t_const = 0
    if 2 * d_accel < total_distance:
        # Full trapezoidal profile
        d_const = total_distance - 2 * d_accel # distance of constant speed
        t_const = d_const / max_velocity # time in constant speed
        t_total = 2 * t_accel + t_const
    else:
        # Triangular profile (not enough distance for max velocity)
        # under constanst accel: distance = 1/2*a*t^2 (each acceleration phase is 1/2 of distance)
        t_accel = np.sqrt(total_distance / acceleration) 
        t_total = 2 * t_accel
        max_velocity = acceleration * t_accel  # Adjusted max velocity

    t = np.arange(0, t_total + dt, dt)

    v = np.piecewise(t,
                    [t < t_accel,
                    (t >= t_accel) & (t < t_accel + t_const),
                    t >= t_accel + t_const],
                    [lambda t: acceleration * t,
                    lambda _: max_velocity,
                    lambda t: max_velocity - acceleration * (t - (t_accel + t_const))])

    z_values = direction * cumtrapz(v, t, initial=0) + start_z

    # Ensure the last value is exactly end_z
    if z_values[-1] != end_z:
        z_values = np.append(z_values, end_z)

    pose_traj = [RigidTransform(rotation=default_rotation,
                                translation=[x, y, z],
                                from_frame='franka_tool',
                                to_frame='world') for z in z_values]

    T = len(pose_traj) * dt
    return pose_traj, dt, T

def get_pre_place_pickup_joints(package_share_directory, location):
    """
    Function to get the joint angles that correspond to the pre-pickup or pre-place position
    """
    traj_file_path = get_traj_file(package_share_directory, location, "home")
    if traj_file_path is None:
        raise Exception("Invalid location provided...")
    with open(traj_file_path, 'rb') as pkl_f:
        skill_data = pickle.load(pkl_f)

    assert skill_data[0]['skill_description'] == 'GuideMode', \
        "Trajectory not collected in guide mode"
    skill_state_dict = skill_data[0]['skill_state_dict']

    joints = skill_state_dict['q'][0]
    return joints

def get_traj_file(package_share_directory, curr_location, end_location):
    '''
    Returns the trajectory pkl file based on the current and desired end location

    Inputs:
        package_share_directory: location of share directory
        curr_location: current arm location
        end_location: desired end location
    
    Outputs:
        traj_file_path: complete file path to .pkl file
    '''
    pkl_file_name = None
    traj_id = TRAJECTORY_ID_MAP[curr_location][end_location]
    if traj_id in TRAJECTORY_FILE_MAP:
        pkl_file_name = TRAJECTORY_FILE_MAP[traj_id]
    else:
        return None
    
    traj_file_path = os.path.join(package_share_directory, pkl_file_name)
    return traj_file_path


def convert_to_float(d):
    return {key: float(value) for key, value in d.items()}

def save_offsets_to_yaml(bin_offsets, assembly_offset):
    config_file = os.path.expanduser(
        '~/Documents/manipulation_ws/src/snaak_manipulation/config/offsets.yaml'
    )

# Convert the dictionaries to ensure float values
    bin_offset_float = convert_to_float(bin_offsets)

    # Updated config dictionary with float values
    updated_config = {
        'snaak_manipulation': {
            'ros__parameters': {
                'bin_end_effector_offsets': bin_offset_float,  # converted to float
                'assembly_end_effector_offset': float(assembly_offset)  # converted to float
            }
        }
    }
    with open(config_file, 'w') as f:
        yaml.dump(updated_config, f)
