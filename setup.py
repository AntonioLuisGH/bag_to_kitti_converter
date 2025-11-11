from setuptools import find_packages, setup

package_name = 'bag_to_kitti_converter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # FIXED: This line had a typo ('T' instead of '+')
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@todo.com',
    description='Converts ROS2 Bag to KITTI format for PointPillars',
    license='Apache-2.0',
    # FIXED: 'tests_require' is deprecated. This is the modern way.
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'bag_to_velodyne = bag_to_kitti_converter.bag_to_velodyne:main',
            'bag_to_labels = bag_to_kitti_converter.bag_to_labels:main',
        ],
    },
)