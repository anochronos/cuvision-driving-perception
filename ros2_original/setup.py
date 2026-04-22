from setuptools import find_packages, setup

package_name = 'controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'cv_bridge',
        'torch',
        'ultralytics',
    ],
    zip_safe=True,
    maintainer='omar',
    maintainer_email='omar@todo.todo',
    description='ROS2 perception pipeline with lane detection and traffic sign recognition',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'tcp_receiver_node = controller.tcp_receiver_node:main',
            'lane_detection_node = controller.lane_detection_node:main',
            'object_detection_node = controller.object_detection_node:main',
            'visualization_node = controller.visualization_node:main',
            'video_publisher = controller.video_publisher:main',
        ],
    },
)
