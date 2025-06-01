from setuptools import setup

package_name = 'base_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools', 'numpy<2.0.0', 'opencv-python', 'torch', 'ultralytics'],
    include_package_data=True,
    package_data={'base_detection': ['best.pt']},
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'base_detection/best.pt']),
        ('share/' + package_name + '/launch', ['launch/base_detection.launch.py']),
    ],
    maintainer='lufa',
    maintainer_email='lufa@todo.todo',
    description='The base_detection package for ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'base_detection = base_detection.base_detection:main',
            'coordinate_receiver = base_detection.coordinate_receiver:main',
            'coordinate_processor = base_detection.coordinate_processor:main',
            'shutown_client = base_detection.shutdown:main' 
        ],
    },
)