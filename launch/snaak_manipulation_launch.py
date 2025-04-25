import launch
import launch_ros.actions
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_file = os.path.expanduser(
        '~/Documents/manipulation_ws/src/snaak_manipulation/config/offsets.yaml'
    )

    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='snaak_pneumatic',
            executable='snaak_pneumatic_control',
            name='snaak_pneumatic_control'),

        launch_ros.actions.Node(
            package='snaak_manipulation',
            executable='snaak_manipulation_node.py',
            name='snaak_manipulation',
            parameters=[config_file])
            
    ])