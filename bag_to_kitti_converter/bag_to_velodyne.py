#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage # Necessary to read /tf_static

import os
import argparse
import struct
import numpy as np
import sys

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix.
    Pure numpy implementation to avoid tf_transformations dependency issues.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
    ])

class BagToVelodyne(Node):
    def __init__(self, bag_path, output_path, topic_name):
        super().__init__('bag_to_velodyne')
        self.bag_path = bag_path
        self.output_path = output_path
        self.topic_name = topic_name
        self.tf_topic_name = '/tf_static'

        # Define frames to look for
        self.target_frame = 'drone_world'
        self.source_frame = 'lidar_link'

        self.velodyne_dir = os.path.join(self.output_path, 'velodyne')
        os.makedirs(self.velodyne_dir, exist_ok=True)
        
        self.timestamp_file = os.path.join(self.output_path, 'timestamps.csv')

        # Initialize Rotation Matrix as Identity (default if no TF found)
        self.R_GLOBAL = np.eye(3)

    def get_static_transform(self):
        """
        Pass 1: Scan the bag solely for /tf_static to get the rotation matrix.
        """
        self.get_logger().info(f"Scanning for static transform on topic: {self.tf_topic_name}...")
        
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        if self.tf_topic_name not in type_map:
            self.get_logger().warn(f"Topic '{self.tf_topic_name}' not found. Using Identity matrix.")
            return

        # Filter only for tf_static
        storage_filter = rosbag2_py.StorageFilter(topics=[self.tf_topic_name])
        reader.set_filter(storage_filter)

        tf_msg_type = get_message(type_map[self.tf_topic_name])
        found_transform = False

        while reader.has_next() and not found_transform:
            (topic, data, timestamp_ns) = reader.read_next()
            
            if topic == self.tf_topic_name:
                msg = deserialize_message(data, tf_msg_type)
                
                # Check all transforms in the list
                for t in msg.transforms:
                    # Check if this is the relationship we want (drone_world <-> lidar_link)
                    if t.header.frame_id == self.target_frame and t.child_frame_id == self.source_frame:
                        
                        q_x = t.transform.rotation.x
                        q_y = t.transform.rotation.y
                        q_z = t.transform.rotation.z
                        q_w = t.transform.rotation.w
                        
                        self.get_logger().info(f"Found transform! Rotation: x={q_x}, y={q_y}, z={q_z}, w={q_w}")
                        
                        # Convert to Matrix
                        self.R_GLOBAL = quaternion_to_rotation_matrix([q_x, q_y, q_z, q_w])
                        found_transform = True
                        break
        
        if not found_transform:
            self.get_logger().warn(f"Transform between {self.target_frame} and {self.source_frame} not found. Using Identity.")

    def run(self):
        self.get_logger().info(f"Opening bag file: {self.bag_path}")
        
        # --- STEP 1: Fetch Transform ---
        self.get_static_transform()

        # --- STEP 2: Process Pointclouds ---
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

        self.get_logger().info(f"Processing pointclouds from: {self.topic_name}")
        self.get_logger().info(f"Saving .bin files to: {self.velodyne_dir}")

        try:
            with open(self.timestamp_file, 'w') as ts_file:
                ts_file.write("frame_id,timestamp_ns\n") 

                while reader.has_next():
                    (topic, data, timestamp_ns) = reader.read_next()
                    
                    if topic == self.topic_name:
                        ros_msg = deserialize_message(data, msg_type)
                        output_filename = os.path.join(self.velodyne_dir, f"{frame_id:06d}.bin")
                        
                        try:
                            points = list(pc2.read_points(ros_msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
                        except Exception as e:
                            # Fallback if intensity missing
                            points_no_intensity = pc2.read_points(ros_msg, field_names=('x', 'y', 'z'), skip_nans=True)
                            points = list(list(p) + [1.0] for p in points_no_intensity)
                    
                        # --- FILTERING AND TRANSFORM ---

                        # 1. Filter out bad points
                        good_points = []
                        for p in points:
                            try:
                                x, y, z, intensity = p[0], p[1], p[2], p[3]
                                good_points.append([x, y, z, intensity])
                            except (IndexError, TypeError):
                                continue

                        # 2. Transform using the Dynamic R_GLOBAL found in Step 1
                        if len(good_points) == 0:
                            points_transformed = np.empty((0, 3))
                        else:
                            points_np = np.array([p[:3] for p in good_points])
                            # Apply the rotation matrix we found earlier
                            points_transformed = np.dot(self.R_GLOBAL, points_np.T).T

                        # 3. Write to file
                        with open(output_filename, 'wb') as f:
                            for i in range(len(points_transformed)):
                                p_transformed = points_transformed[i]
                                intensity = good_points[i][3] 
                                
                                try:
                                    f.write(struct.pack('ffff', p_transformed[0], p_transformed[1], p_transformed[2], intensity))
                                except IndexError:
                                    continue
                        
                        ts_file.write(f"{frame_id},{timestamp_ns}\n")

                        if frame_id % 10 == 0:
                            self.get_logger().info(f"Processed frame {frame_id}")
                        
                        frame_id += 1
        except Exception as e:
            self.get_logger().error(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.get_logger().info(f"Finished processing. Total frames: {frame_id}")


def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Convert ROS bag PointCloud2 to KITTI .bin files.')
    parser.add_argument('--bag-path', type=str, required=True, help='Path to the ROS bag file')
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