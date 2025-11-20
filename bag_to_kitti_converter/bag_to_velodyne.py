#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage

import os
import argparse
import struct
import numpy as np
import sys
import bisect

# --- HELPER: Pure Numpy Quaternion to Matrix ---
def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
    ])

class TFManager:
    """
    Helper class to manage changing static transforms in a mixed bag.
    """
    def __init__(self, bag_path, tf_topic, target_frame, source_frame, logger):
        self.bag_path = bag_path
        self.tf_topic = tf_topic
        self.target_frame = target_frame
        self.source_frame = source_frame
        self.logger = logger
        
        # List of tuples: (timestamp_ns, rotation_matrix_3x3)
        self.transforms = [] 
        
        self._load_transforms()

    def _load_transforms(self):
        """
        Scans the entire bag for ALL /tf_static messages and stores them.
        """
        self.logger.info(f"Scanning {self.tf_topic} for all transform changes...")
        
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        if self.tf_topic not in type_map:
            self.logger.warn(f"Topic {self.tf_topic} not found. Using Identity.")
            return

        storage_filter = rosbag2_py.StorageFilter(topics=[self.tf_topic])
        reader.set_filter(storage_filter)
        msg_type = get_message(type_map[self.tf_topic])

        count = 0
        while reader.has_next():
            (topic, data, timestamp_ns) = reader.read_next()
            msg = deserialize_message(data, msg_type)
            
            for t in msg.transforms:
                if t.header.frame_id == self.target_frame and t.child_frame_id == self.source_frame:
                    # Extract Quaternion
                    q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
                    # Convert to Matrix
                    R = quaternion_to_rotation_matrix(q)
                    
                    # Use the timestamp from the Message Header, not the bag write time
                    # (This is more accurate for when the transform actually applies)
                    msg_time = t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec
                    
                    self.transforms.append((msg_time, R))
                    count += 1
        
        # Sort by timestamp so we can search efficiently
        self.transforms.sort(key=lambda x: x[0])
        self.logger.info(f"Found {count} static transform updates.")

    def get_matrix_at_time(self, query_time_ns):
        """
        Returns the rotation matrix active at query_time_ns.
        Logic: Find the last transform that happened BEFORE query_time_ns.
        """
        if not self.transforms:
            return np.eye(3) # Default to Identity if nothing found

        # Extract timestamps for binary search
        timestamps = [t[0] for t in self.transforms]
        
        # bisect_right gives the insertion point to keep order. 
        # The element to the LEFT of this point is the valid transform.
        idx = bisect.bisect_right(timestamps, query_time_ns)
        
        if idx == 0:
            # The query time is BEFORE the first transform. 
            # Usually safe to use the first known transform.
            return self.transforms[0][1]
        else:
            # Return the transform that started just before this frame
            return self.transforms[idx - 1][1]


class BagToVelodyne(Node):
    def __init__(self, bag_path, output_path, topic_name):
        super().__init__('bag_to_velodyne')
        self.bag_path = bag_path
        self.output_path = output_path
        self.topic_name = topic_name
        
        # Configure Frames
        self.tf_topic = '/tf_static' 
        self.target_frame = 'drone_world'
        self.source_frame = 'lidar_link'

        self.velodyne_dir = os.path.join(self.output_path, 'velodyne')
        os.makedirs(self.velodyne_dir, exist_ok=True)
        self.timestamp_file = os.path.join(self.output_path, 'timestamps.csv')

        # --- 1. Initialize Transform Manager ---
        self.tf_manager = TFManager(
            self.bag_path, 
            self.tf_topic, 
            self.target_frame, 
            self.source_frame, 
            self.get_logger()
        )

    def run(self):
        self.get_logger().info(f"Processing pointclouds...")
        
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        if self.topic_name not in type_map:
            self.get_logger().error(f"Topic '{self.topic_name}' not found.")
            return

        storage_filter = rosbag2_py.StorageFilter(topics=[self.topic_name])
        reader.set_filter(storage_filter)

        frame_id = 0
        msg_type = get_message(type_map[self.topic_name])

        try:
            with open(self.timestamp_file, 'w') as ts_file:
                ts_file.write("frame_id,timestamp_ns\n") 

                while reader.has_next():
                    (topic, data, timestamp_ns) = reader.read_next()
                    
                    if topic == self.topic_name:
                        ros_msg = deserialize_message(data, msg_type)
                        output_filename = os.path.join(self.velodyne_dir, f"{frame_id:06d}.bin")
                        
                        # --- 2. GET DYNAMIC ROTATION FOR THIS FRAME ---
                        # We ask the manager: "What was the rotation at this timestamp?"
                        R_CURRENT = self.tf_manager.get_matrix_at_time(timestamp_ns)

                        # Read Points
                        try:
                            points = list(pc2.read_points(ros_msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
                        except Exception:
                            points_no_intensity = pc2.read_points(ros_msg, field_names=('x', 'y', 'z'), skip_nans=True)
                            points = list(list(p) + [1.0] for p in points_no_intensity)
                    
                        # Filter bad points
                        good_points = []
                        for p in points:
                            try:
                                x, y, z, i = p[0], p[1], p[2], p[3]
                                good_points.append([x, y, z, i])
                            except: continue

                        # --- 3. APPLY ROTATION ---
                        if len(good_points) == 0:
                            points_transformed = np.empty((0, 3))
                        else:
                            points_np = np.array([p[:3] for p in good_points])
                            # Apply the matrix specific to this frame
                            points_transformed = np.dot(R_CURRENT, points_np.T).T

                        # Write to .bin
                        with open(output_filename, 'wb') as f:
                            for i in range(len(points_transformed)):
                                p_t = points_transformed[i]
                                intensity = good_points[i][3] 
                                f.write(struct.pack('ffff', p_t[0], p_t[1], p_t[2], intensity))
                        
                        ts_file.write(f"{frame_id},{timestamp_ns}\n")

                        if frame_id % 50 == 0:
                            self.get_logger().info(f"Frame {frame_id}: Applied TF from time {timestamp_ns}")
                        
                        frame_id += 1
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.get_logger().info(f"Finished. Total frames: {frame_id}")


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--topic', type=str, default='/lidar/avia')
    
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