"""Launch all 4 nodes of the autonomous driving pipeline."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('driving_pipeline'), 'config', 'pipeline.yaml')

    return LaunchDescription([
        Node(
            package='driving_pipeline',
            executable='tcp_camera_node',
            name='tcp_camera_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='driving_pipeline',
            executable='tlnp_node',
            name='tlnp_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='driving_pipeline',
            executable='detection_node',
            name='detection_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='driving_pipeline',
            executable='viz_node',
            name='viz_node',
            parameters=[config],
            output='screen',
        ),
    ])
