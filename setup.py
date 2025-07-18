from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'base_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=['base_detection'],
    install_requires=['setuptools', 'numpy<2.0.0', 'opencv-python', 'torch', 'ultralytics', 'scipy', 'matplotlib', 'scikit-learn'],
    include_package_data=True,
    package_data={'': ['best.pt']},
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    maintainer='Pedro Saraiva',
    maintainer_email='pedro.a.saraiva08@gmail.com',
    description='Base detection package for UAV precision landing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'base_detector = base_detection.base_detection:main',
            'coordinate_receiver = base_detection.coordinate_receiver:main',
            'coordinate_processor = base_detection.coordinate_processor:main',
            'package_detector_node = base_detection.package_detector_node:main'
        ],
    },
)