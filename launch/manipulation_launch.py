import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='snaak_pneumatic',
            executable='snaak_pneumatic_control',
            name='pneumatic_control'),

        launch_ros.actions.Node(
            package='snaak_manipulation',
            executable='manipulation_node.py',
            name='manipulation_action_server'),
    ])