import numpy as np
from scipy.integrate import cumtrapz
from autolab_core import RigidTransform
from snaak_manipulation_constants import JOINTS_MAP, BIN_OFFSETS
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from frankapy import utils

def get_traj(q1, q2, dt=0.01, T=5.0):
    '''
    Generates a joint trajectory from q1 to q2 using a minimum jerk profile.
    '''
    ts = np.arange(0, T, dt)
    q1 = np.array(q1)
    q2 = np.array(q2)
    joints_traj = [utils.min_jerk(q1, q2, t, T) for t in ts]
    return joints_traj, T, dt
        
        

def get_bin_offset(bin_id):
    '''
    Return XYZ offset from arm origin to center of bin
    '''
    return BIN_OFFSETS[bin_id]


def pickup_traj(x, y, start_z, end_z, default_rotation, step_size=0.001, acceleration = 0.1):
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

def get_joints(location):
    """
    Function to get the joint angles that correspond to the pre-pickup or pre-place position
    """
    if location not in JOINTS_MAP:
        raise Exception("Invalid location provided...")
    desired_joints = JOINTS_MAP[location]
    desired_joints = np.array(desired_joints)
    return desired_joints

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
