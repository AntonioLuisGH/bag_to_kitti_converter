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
import sys
import tf_transformations # --- NEW ---

# --- NEW: Define the global transform to move the whole scene ---
# This is a 180-degree rotation around the Z-axis (yaw).
# It will flip the X and Y axes.
R_GLOBAL = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])


class BagToVelodyne(Node):
    def __init__(self, bag_path, output_path, topic_name):
        super().__init__('bag_to_velodyne')
        self.bag_path = bag_path
        self.output_path = output_path
        self.topic_name = topic_name

        self.velodyne_dir = os.path.join(self.output_path, 'velodyne')
        os.makedirs(self.velodyne_dir, exist_ok=True)
        
        self.timestamp_file = os.path.join(self.output_path, 'timestamps.csv')

    def run(self):
        self.get_logger().info(f"Opening bag file: {self.bag_path}")
        
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        if self.topic_name not in type_map:
            self.get_logger().error(f"Topic '{self.topic_name}' not found in bag file.")
            return

        storage_filter = rosbag2_py.StorageFilter(topics=[self.topic_name])
        reader.set_filter(storage_filter)

        frame_id = 0
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
                        ros_msg = deserialize_message(data, msg_type)
                        output_filename = os.path.join(self.velodyne_dir, f"{frame_id:06d}.bin")
                        
                        try:
                            points = list(pc2.read_points(ros_msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
                        except Exception as e:
                            self.get_logger().warn(f"Could not read 'intensity' field, defaulting to 1.0. Error: {e}")
                            points_no_intensity = pc2.read_points(ros_msg, field_names=('x', 'y', 'z'), skip_nans=True)
                            points = list(list(p) + [1.0] for p in points_no_intensity) # <-- Fixed list()
                    
                        # --- START: NEW FILTERING AND TRANSFORM ---

                        # 1. Filter out "bad" points (which pc2 returns as 0-dim arrays)
                        good_points = []
                        for p in points:
                            try:
                                # Try to access the first 4 elements
                                # This will fail for 0-dimensional arrays
                                x, y, z, intensity = p[0], p[1], p[2], p[3]
                                good_points.append([x, y, z, intensity])
                            except (IndexError, TypeError):
                                # Skip bad points
                                continue

                        # 2. Transform only the good points
                        if len(good_points) == 0:
                            points_transformed = np.empty((0, 3))
                        else:
                            points_np = np.array([p[:3] for p in good_points])
                            points_transformed = np.dot(R_GLOBAL, points_np.T).T

                        # --- END: NEW FILTERING AND TRANSFORM ---

                        # 3. Write the transformed points (and original intensity) to file
                        with open(output_filename, 'wb') as f:
                            for i in range(len(points_transformed)):
                                # Get the transformed (x, y, z)
                                p_transformed = points_transformed[i]
                                # Get the original intensity from the good_points list
                                intensity = good_points[i][3] 
                                
                                try:
                                    f.write(struct.pack('ffff', p_transformed[0], p_transformed[1], p_transformed[2], intensity))
                                except IndexError:
                                    # This should no longer happen, but good to keep
                                    self.get_logger().error("Point did not have 4 values. Skipping.")
                                    continue
                        
                        ts_file.write(f"{frame_id},{timestamp_ns}\n")

                        if frame_id % 10 == 0:
                            self.get_logger().info(f"Processed frame {frame_id} (Timestamp: {timestamp_ns})")
                        
                        frame_id += 1
        except Exception as e:
            self.get_logger().error(f"An error occurred: {e}")
            import traceback # Add this
            traceback.print_exc() # Add this to see the exact line
        finally:
            self.get_logger().info(f"Finished processing. Total frames: {frame_id}")


def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Convert ROS bag PointCloud2 to KITTI .bin files.')
    parser.add_argument('--bag-path', type=str, required=True, help='Path to the ROS bag file (e.g., /path/to/bag.db3)')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output KITTI directory')
    parser.add_argument('--topic', type=str, default='/lidar/avia', help='The PointCloud2 topic name')
    
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