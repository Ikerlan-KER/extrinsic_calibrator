import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'extrinsic_calibrator_examples'
    rviz_dir = os.path.join(get_package_share_directory(package_name), 'rviz')

    rviz_config = os.path.join(rviz_dir, 'extrinsic.rviz')

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        ),

    ])
