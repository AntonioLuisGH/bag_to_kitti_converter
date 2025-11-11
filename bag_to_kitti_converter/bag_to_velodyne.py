#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

import os
import argparse
import struct
import numpy as np
import sys # <-- Import sys

# FIXED: Removed 4-space indentation from the entire file
class BagToVelodyne(Node):
    def __init__(self, bag_path, output_path, topic_name):
        super().__init__('bag_to_velodyne')
        self.bag_path = bag_path
        self.output_path = output_path
        self.topic_name = topic_name

        # Create output directories
        self.velodyne_dir = os.path.join(self.output_path, 'velodyne')
        os.makedirs(self.velodyne_dir, exist_ok=True)
        
        # Timestamp file
        self.timestamp_file = os.path.join(self.output_path, 'timestamps.csv')

    def run(self):
        self.get_logger().info(f"Opening bag file: {self.bag_path}")
        
        # Setup bag reader
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)

        # Get topic types
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        if self.topic_name not in type_map:
            self.get_logger().error(f"Topic '{self.topic_name}' not found in bag file.")
            self.get_logger().error(f"Available topics: {list(type_map.keys())}")
            return

        # Set filter for the lidar topic
        storage_filter = rosbag2_py.StorageFilter(topics=[self.topic_name])
        reader.set_filter(storage_filter)

        frame_id = 0
        
        # Get message definition
        msg_type = get_message(type_map[self.topic_name])

        self.get_logger().info(f"Processing topic: {self.topic_name}")
        self.get_logger().info(f"Saving .bin files to: {self.velodyne_dir}")
        self.get_logger().info(f"Saving timestamps to: {self.timestamp_file}")

        try:
            with open(self.timestamp_file, 'w') as ts_file:
                ts_file.write("frame_id,timestamp_ns\n") # CSV Header

                while reader.has_next():
                    (topic, data, timestamp_ns) = reader.read_next()
                    
                    if topic == self.topic_name:
                        # Deserialize message
                        ros_msg = deserialize_message(data, msg_type)
                        
                        # Prepare file path
                        output_filename = os.path.join(self.velodyne_dir, f"{frame_id:06d}.bin")
                        
                        # Use sensor_msgs_py to read points
                        # We assume fields 'x', 'y', 'z', and 'intensity' exist.
                        try:
                            points = pc2.read_points(ros_msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True)
                        except Exception as e:
                            self.get_logger().warn(f"Could not read 'intensity' field, defaulting to 1.0. Error: {e}")
                            points_no_intensity = pc2.read_points(ros_msg, field_names=('x', 'y', 'z'), skip_nans=True)
                            # Add a default intensity
                            points = (list(p) + [1.0] for p in points_no_intensity)

                        # Write to binary file
                        with open(output_filename, 'wb') as f:
                            for p in points:
                                # p[0]=x, p[1]=y, p[2]=z, p[3]=intensity
                                # Pack as 4 floats (float32)
                                try:
                                    f.write(struct.pack('ffff', p[0], p[1], p[2], p[3]))
                                except IndexError:
                                    self.get_logger().error("Point did not have 4 values. Skipping.")
                                    continue
                        
                        # Write to timestamp file
                        ts_file.write(f"{frame_id},{timestamp_ns}\n")

                        if frame_id % 10 == 0:
                            self.get_logger().info(f"Processed frame {frame_id} (Timestamp: {timestamp_ns})")
                        
                        frame_id += 1

        except Exception as e:
            self.get_logger().error(f"An error occurred: {e}")
        finally:
            self.get_logger().info(f"Finished processing. Total frames: {frame_id}")


def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Convert ROS bag PointCloud2 to KITTI .bin files.')
    parser.add_argument('--bag-path', type=str, required=True, help='Path to the ROS bag file (e.g., /path/to/bag.db3)')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output KITTI directory')
    parser.add_argument('--topic', type=str, default='/lidar/avia', help='The PointCloud2 topic name')
    
    # FIXED: Explicitly pass sys.argv[1:] to remove_ros_args
    # This prevents the script name (sys.argv[0]) from being passed to argparse
    parsed_args = parser.parse_args(args=rclpy.utilities.remove_ros_args(args=sys.argv[1:]))

    node = BagToVelodyne(
        bag_path=parsed_args.bag_path,
        output_path=parsed_args.output_path,
        topic_name=parsed_args.topic
    )
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()