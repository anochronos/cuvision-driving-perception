from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'driving_pipeline'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'tcp_camera_node = driving_pipeline.tcp_camera_node:main',
            'tlnp_node = driving_pipeline.tlnp_node:main',
            'detection_node = driving_pipeline.detection_node:main',
            'viz_node = driving_pipeline.viz_node:main',
        ],
    },
)
