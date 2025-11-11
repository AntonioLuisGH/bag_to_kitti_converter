#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py
from visualization_msgs.msg import Marker
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

import os
import argparse
import csv
import numpy as np
import tf_transformations
import bisect
import sys # <-- Import sys

# FIXED: Removed 4-space indentation from the entire file
class BagToLabels(Node):
    def __init__(self, bag_path, output_path, bbox_topic, tf_topic):
        super().__init__('bag_to_labels')
        self.bag_path = bag_path
        self.output_path = output_path
        self.bbox_topic = bbox_topic
        self.tf_topic = tf_topic

        # Input timestamp file
        self.timestamp_file = os.path.join(self.output_path, 'timestamps.csv')
        
        # Output directory
        self.labels_dir = os.path.join(self.output_path, 'label_2')
        os.makedirs(self.labels_dir, exist_ok=True)

        self.static_transform = None
        self.bbox_messages = [] # List of (timestamp, msg)
    
    def get_bag_reader(self):
        """Helper to create a new bag reader instance."""
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)
        return reader

    def get_topic_map(self):
        """Helper to get the topic name to type map."""
        reader = self.get_bag_reader()
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        # Explicitly close reader
        del reader
        return type_map

    def find_static_transform(self, type_map):
        """Iterate through the bag to find the static TF."""
        self.get_logger().info(f"Searching for static transform on topic: {self.tf_topic}")
        reader = self.get_bag_reader()
        storage_filter = rosbag2_py.StorageFilter(topics=[self.tf_topic])
        reader.set_filter(storage_filter)
        
        msg_type = get_message(type_map[self.tf_topic])

        while reader.has_next():
            (topic, data, _) = reader.read_next()
            if topic == self.tf_topic:
                tf_msg = deserialize_message(data, msg_type)
                for transform in tf_msg.transforms:
                    # Find the transform from drone_world to lidar_link
                    if transform.header.frame_id == 'drone_world' and transform.child_frame_id == 'lidar_link':
                        self.static_transform = transform.transform # Store the Transform message
                        self.get_logger().info(f"Found static transform: drone_world -> lidar_link")
                        del reader # Close reader
                        return
        
        del reader # Close reader
        self.get_logger().warn("Could not find the 'drone_world' to 'lidar_link' static transform.")

    def cache_bbox_messages(self, type_map):
        """Iterate through the bag to cache all bounding box messages."""
        self.get_logger().info(f"Caching all messages from: {self.bbox_topic}")
        reader = self.get_bag_reader()
        storage_filter = rosbag2_py.StorageFilter(topics=[self.bbox_topic])
        reader.set_filter(storage_filter)

        msg_type = get_message(type_map[self.bbox_topic])

        while reader.has_next():
            (topic, data, timestamp_ns) = reader.read_next()
            if topic == self.bbox_topic:
                bbox_msg = deserialize_message(data, msg_type)
                self.bbox_messages.append((timestamp_ns, bbox_msg))
        
        # Sort by timestamp for fast lookup
        self.bbox_messages.sort(key=lambda x: x[0])
        self.get_logger().info(f"Cached {len(self.bbox_messages)} bounding box messages.")
        del reader

    def find_closest_bbox(self, target_timestamp_ns):
        """Finds the bbox message with the timestamp closest to the target."""
        if not self.bbox_messages:
            return None
        
        # Get just the timestamps
        timestamps = [t[0] for t in self.bbox_messages]
        
        # Find insertion point
        idx = bisect.bisect_left(timestamps, target_timestamp_ns)
        
        if idx == 0:
            # Closest is the first element
            return self.bbox_messages[0][1]
        if idx == len(timestamps):
            # Closest is the last element
            return self.bbox_messages[-1][1]
        
        # Check timestamps at idx and idx-1
        ts_before = timestamps[idx - 1]
        ts_after = timestamps[idx]
        
        if (target_timestamp_ns - ts_before) < (ts_after - target_timestamp_ns):
            return self.bbox_messages[idx - 1][1]
        else:
            return self.bbox_messages[idx][1]

    def process_labels(self):
        """Read timestamps.csv and generate KITTI labels."""
        self.get_logger().info(f"Processing labels based on: {self.timestamp_file}")
        self.get_logger().info(f"Saving .txt files to: {self.labels_dir}")
        
        if not os.path.exists(self.timestamp_file):
            self.get_logger().error(f"Timestamp file not found: {self.timestamp_file}")
            self.get_logger().error("Please run the bag_to_velodyne node first!")
            return

        if not self.static_transform:
            self.get_logger().error("Static transform not found. Cannot process labels.")
            return

        # Prepare static transform components
        static_trans = np.array([
            self.static_transform.translation.x,
            self.static_transform.translation.y,
            self.static_transform.translation.z
        ])
        static_rot_quat = np.array([
            self.static_transform.rotation.x,
            self.static_transform.rotation.y,
            self.static_transform.rotation.z,
            self.static_transform.rotation.w
        ])
        # Get 3x3 rotation matrix from the static transform
        static_rot_matrix = tf_transformations.quaternion_matrix(static_rot_quat)[:3, :3]

        with open(self.timestamp_file, 'r') as ts_file:
            reader = csv.reader(ts_file)
            next(reader) # Skip header
            
            for row in reader:
                frame_id = int(row[0])
                timestamp_ns = int(row[1])
                
                # Find the closest bounding box message
                marker_msg = self.find_closest_bbox(timestamp_ns)
                
                if not marker_msg:
                    self.get_logger().warn(f"No bbox message found for frame {frame_id}. Skipping.")
                    continue
                
                # Extract marker data (in drone_world frame)
                marker_pos = np.array([
                    marker_msg.pose.position.x,
                    marker_msg.pose.position.y,
                    marker_msg.pose.position.z
                ])
                marker_quat = np.array([
                    marker_msg.pose.orientation.x,
                    marker_msg.pose.orientation.y,
                    marker_msg.pose.orientation.z,
                    marker_msg.pose.orientation.w
                ])
                
                # --- Apply Static Transform ---
                # Transform position: Rot(static) * Pos(marker) + Trans(static)
                transformed_pos = np.dot(static_rot_matrix, marker_pos) + static_trans
                
                # Transform orientation: Quat(static) * Quat(marker)
                transformed_quat = tf_transformations.quaternion_multiply(static_rot_quat, marker_quat)
                
                # --- Format for KITTI ---
                
                # object_type
                object_type = "Drone" # As requested, only one category
                
                # truncation, occlusion, alpha, 2D box (placeholders)
                truncation = 0.0
                occlusion = 0 # 0=fully visible
                alpha = 0.0 # Observation angle
                left, top, right, bottom = 0.0, 0.0, 0.0, 0.0 # 2D box
                
                # 3D dimensions (h, w, l)
                # NOTE: Marker scale (x,y,z) maps to KITTI (l,w,h)
                h = marker_msg.scale.z
                w = marker_msg.scale.y
                l = marker_msg.scale.x
                
                # 3D location (x, y, z)
                # This is the center of the box in the lidar_link frame
                x = transformed_pos[0]
                y = transformed_pos[1]
                z_center = transformed_pos[2]
                
                # ** IMPORTANT KITTI ADJUSTMENT **
                # KITTI (x,y,z) is the *bottom center* of the box,
                # but the marker's pose is its *geometric center*.
                # We must adjust 'z' by half the height.
                z = z_center # - h / 2.0  <-- This is the typical adjustment
                # The user's TF has lidar at z=0.5 relative to world.
                # The marker pose is likely also relative to world.
                # The transformed_pos is now relative to lidar_link.
                # If z=0 is the ground in lidar_link, then z=z_center - h/2.0 is correct.
                # If the lidar is 0.5m *above* the ground, then the z coord is fine as is
                # relative to the lidar, but the KITTI format wants z relative to
                # the *camera* (or in our case, lidar) frame's origin.
                # Let's stick to the standard definition: (x,y,z) is the box center.
                # If PointPillars expects bottom-center, you will need to adjust:
                # z = z_center - (h / 2.0)
                # For now, I will provide the center coordinates, as this is safer.
                # --> REVISITING: The user example shows `y` (vertical) coord, not z.
                # KITTI: (x,y,z) in *camera* coordinates. x=right, y=down, z=forward.
                # Lidar: (x,y,z) in *lidar* coordinates. x=forward, y=left, z=up.
                # We must convert!
                # Lidar (x,y,z) -> KITTI (z, -x, -y)
                # But our lidar_link is x=forward, y=left, z=up (ROS std)
                # KITTI velodyne is x=forward, y=left, z=up
                # The user's PointPillars setup might expect ROS-style lidar coords.
                # Let's provide the coordinates in the `lidar_link` frame directly,
                # as this is the most common for lidar-based detectors.
                # x_kitti, y_kitti, z_kitti = transformed_pos[0], transformed_pos[1], transformed_pos[2]
                
                # --> Final Decision: The example shows (x,y,z) which are (loc_x, loc_y, loc_z)
                # in the *camera* frame. The user is providing data in the *lidar* frame.
                # Most PointPillars implementations *expect* data in the lidar (velodyne) frame.
                # The (x,y,z) in the label should be the (x,y,z) *in the lidar frame*.
                # BUT, KITTI label 'y' is the vertical coordinate *relative to the ground*.
                # The marker's 'z' is its center. `transformed_pos[2]` is the center 'z' in lidar_link.
                # The example label for y is ~1.7m. This is likely the *center* of the car.
                # I will provide the 3D location (x,y,z) in the `lidar_link` frame.
                # The user must ensure their PointPillars config uses this frame.
                # x_loc = transformed_pos[0]
                # y_loc = transformed_pos[1]
                # z_loc = transformed_pos[2] # This is z_center
                
                # Per KITTI format: (h, w, l) and (x, y, z, rotation_y)
                # (x, y, z) is location of bottom-center in *camera* coords.
                # Our (x,y,z) is center in *lidar* coords.
                # Let's assume the user's PointPillars takes lidar coords.
                # We just need to adjust z to be bottom-center.
                x_loc = transformed_pos[0]
                y_loc = transformed_pos[1]
                z_loc = transformed_pos[2] - (h / 2.0)
                
                # rotation_y
                # We need the yaw from the transformed orientation
                (_, _, yaw) = tf_transformations.euler_from_quaternion(transformed_quat)
                rotation_y = yaw
                
                # Create the KITTI label line
                kitti_line = (
                    f"{object_type} {truncation:.2f} {occlusion} {alpha:.2f} "
                    f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
                    f"{h:.2f} {w:.2f} {l:.2f} "
                    f"{x_loc:.2f} {y_loc:.2f} {z_loc:.2f} {rotation_y:.2f}"
                )
                
                # Write to file
                output_filename = os.path.join(self.labels_dir, f"{frame_id:06d}.txt")
                with open(output_filename, 'w') as f:
                    f.write(kitti_line + '\n')
                
                if frame_id % 10 == 0:
                    self.get_logger().info(f"Processed label for frame {frame_id}")
                    
        self.get_logger().info("Finished processing all labels.")

    def run(self):
        # Get topic info first
        type_map = self.get_topic_map()
        if self.bbox_topic not in type_map or self.tf_topic not in type_map:
            self.get_logger().error("Required topics not in bag.")
            if self.bbox_topic not in type_map: self.get_logger().error(f"- Missing: {self.bbox_topic}")
            if self.tf_topic not in type_map: self.get_logger().error(f"- Missing: {self.tf_topic}")
            return
            
        # 1. Find the static transform
        self.find_static_transform(type_map)
        
        # 2. Cache all bbox messages
        self.cache_bbox_messages(type_map)
        
        # 3. Process labels using the timestamp file
        self.process_labels()


def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Convert ROS bag Marker to KITTI label .txt files.')
    parser.add_argument('--bag-path', type=str, required=True, help='Path to the ROS bag file (e.g., /path/to/bag.db3)')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output KITTI directory')
    parser.add_argument('--bbox-topic', type=str, default='/drone_bbox', help='The Marker topic name')
    parser.add_argument('--tf-topic', type=str, default='/tf_static', help='The TF static topic name')
    
    # FIXED: Explicitly pass sys.argv[1:] to remove_ros_args
    # This prevents the script name (sys.argv[0]) from being passed to argparse
    parsed_args = parser.parse_args(args=rclpy.utilities.remove_ros_args(args=sys.argv[1:]))

    node = BagToLabels(
        bag_path=parsed_args.bag_path,
        output_path=parsed_args.output_path,
        bbox_topic=parsed_args.bbox_topic,
        tf_topic=parsed_args.tf_topic
    )
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()