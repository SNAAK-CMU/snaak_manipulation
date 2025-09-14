# Manipulation Actions:

## snaak_manipulation/execute_trajectory

Executes a predefined trajectory between the current location and the location id

 Input:
   - location id (bin1, bin2, bin3, home, assembly)
 Output:
   - robot pose

To call action from command line:
1) unlock arm
2) launch frankapy
3) source manipulation_ws
4) ros2 launch snaak_manipulation snaak_manipulation_launch.py 
5) In another terminal, source manipulation_ws
6) use command:
    ros2 action send_goal /snaak_manipulation/execute_trajectory snaak_manipulation/action/ExecuteTrajectory "{desired_location: bin1}"


## snaak_manipulation/execute_pickup
Performs a pickup motion ending at the specified XYZ. Calls pneumatic service to pick up.
Note: Using impedances, so it is ok/desired if z is lower than the actual ingredient

 Input:
   - x, y, z in robot base frame
   - ingredient type
 Output:
    - robot pose



## snaak_manipulation/execute_place
Performs the place motion using pneumatic eject and disable functions
 Input:
   - x, y, z in robot base frame
   - ingredient type
 Output:
   - robot pose



## snaak_manipulation/return_home
Returns the robot to the home position from the current one

 Input: None
 Output: None
