from setuptools import find_packages, setup

package_name = 'base_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
                      'setuptools',
                      'opencv-python',
                      'numpy',
                      'torch',
                      'ultralytics',
                      'cv_bridge',
                      'message-filters'],
    zip_safe=True,
    maintainer='gustavo',
    maintainer_email='gumotabarros@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'base_detection = base_detection.base_detection:main',
            'px4_gesture_test = base_detection.px4_gesture_test:main',
            'mission_control = base_detection.mission_control:main',
            'gt_odometry = base_detection.gt_odometry:main',
        ],
    },
)
