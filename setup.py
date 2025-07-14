from setuptools import setup

package_name = 'base_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools', 'numpy<2.0.0', 'opencv-python', 'torch', 'ultralytics', 'scipy>=1.15.3', 'matplotlib', 'scikit-learn'],
    include_package_data=True,
    package_data={'base_detection': ['best.pt']},
    zip_safe=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'base_detection/best.pt']),
        ('share/' + package_name + '/launch', ['launch/base_detection.launch.py',"launch/fine_tuning.launch.py"]),
    ],
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Base detection package for UAV precision landing',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'base_detector = base_detection.base_detector:main',
            'coordinate_receiver = base_detection.coordinate_receiver:main',
            'coordinate_processor = base_detection.coordinate_processor:main',
            'shutown_client = base_detection.shutdown:main' 
        ],
    },
)