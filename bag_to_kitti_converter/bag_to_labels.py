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
import sys

# --- NEW: Define the global transform to match the point cloud transformation ---
R_GLOBAL = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])

# --- VIRTUAL KITTI CALIBRATION ---
# R_CAM_LIDAR: Transform from ROS Lidar (x-fwd, y-left, z-up)
#              to KITTI Camera (x-right, y-down, z-fwd)
R_CAM_LIDAR = np.array([
    [ 0, -1,  0], # x_cam = -y_lidar
    [ 0,  0, -1], # y_cam = -z_lidar
    [ 1,  0,  0]  # z_cam =  x_lidar
])
T_CAM_LIDAR = np.array([0, 0, 0]) # Zero translation
# Convert R_CAM_LIDAR to a 4x4 matrix and then a quaternion
# This is the rotation from lidar_link -> camera_0
R_CAM_LIDAR_MAT4 = np.eye(4)
R_CAM_LIDAR_MAT4[:3, :3] = R_CAM_LIDAR
Q_CAM_LIDAR = tf_transformations.quaternion_from_matrix(R_CAM_LIDAR_MAT4)

class BagToLabels(Node):
    def __init__(self, bag_path, output_path, bbox_topic, tf_topic):
        super().__init__('bag_to_labels')
        self.bag_path = bag_path
        self.output_path = output_path
        self.bbox_topic = bbox_topic
        self.tf_topic = tf_topic
        self.timestamp_file = os.path.join(self.output_path, 'timestamps.csv')
        self.labels_dir = os.path.join(self.output_path, 'label_2')
        self.calib_dir = os.path.join(self.output_path, 'calib')
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.calib_dir, exist_ok=True)
        self.static_transform = None # drone_world -> lidar_link
        self.bbox_messages = [] 
    
    def get_bag_reader(self):
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)
        return reader
    
    def get_topic_map(self):
        reader = self.get_bag_reader()
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        del reader
        return type_map
    
    def find_static_transform(self, type_map):
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
                    if transform.header.frame_id == 'drone_world' and transform.child_frame_id == 'lidar_link':
                        self.static_transform = transform.transform
                        self.get_logger().info(f"Found static transform: drone_world -> lidar_link")
                        del reader
                        return
        del reader
        self.get_logger().warn("Could not find the 'drone_world' to 'lidar_link' static transform.")
    
    def cache_bbox_messages(self, type_map):
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
        
        self.bbox_messages.sort(key=lambda x: x[0])
        self.get_logger().info(f"Cached {len(self.bbox_messages)} bounding box messages.")
        del reader
    
    def find_closest_bbox(self, target_timestamp_ns):
        if not self.bbox_messages: return None
        timestamps = [t[0] for t in self.bbox_messages]
        idx = bisect.bisect_left(timestamps, target_timestamp_ns)
        if idx == 0: return self.bbox_messages[0][1]
        if idx == len(timestamps): return self.bbox_messages[-1][1]
        ts_before = timestamps[idx - 1]
        ts_after = timestamps[idx]
        if (target_timestamp_ns - ts_before) < (ts_after - target_timestamp_ns):
            return self.bbox_messages[idx - 1][1]
        else:
            return self.bbox_messages[idx][1]
    
    def write_calib_file(self, frame_id_str):
        """Writes the virtual calibration file for this frame."""
        P_identity = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
        R0_rect = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00"
        Tr_imu = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
        
        tr_data = np.hstack((R_CAM_LIDAR, T_CAM_LIDAR.reshape(3, 1)))
        tr_str = " ".join([f"{x:.6e}" for x in tr_data.flatten()])
        content = f"""P0: {P_identity}
P1: {P_identity}
P2: {P_identity}
P3: {P_identity}
R0_rect: {R0_rect}
Tr_velo_to_cam: {tr_str}
Tr_imu_to_velo: {Tr_imu}
"""
        calib_filename = os.path.join(self.calib_dir, f"{frame_id_str}.txt")
        with open(calib_filename, 'w') as f:
            f.write(content)
    
    def process_labels(self):
        """Read timestamps.csv and generate KITTI labels."""
        self.get_logger().info("Processing labels and calibration files...")
        
        if not os.path.exists(self.timestamp_file):
            self.get_logger().error(f"Timestamp file not found: {self.timestamp_file}")
            return
        if not self.static_transform:
            self.get_logger().error("Static transform not found. Cannot process labels.")
            return
        # --- 1. Get TFs for POSITION (drone_world -> lidar_link) ---
        tf_trans_world_lidar = np.array([ # t
            self.static_transform.translation.x,
            self.static_transform.translation.y,
            self.static_transform.translation.z
        ])
        tf_rot_world_lidar_mat = tf_transformations.quaternion_matrix([ # R
            self.static_transform.rotation.x,
            self.static_transform.rotation.y,
            self.static_transform.rotation.z,
            self.static_transform.rotation.w
        ])[:3, :3]
        
        # We need the INVERSE to move points from world -> lidar
        # P_lidar = R_inv * (P_world - t)
        tf_rot_lidar_world_mat = tf_rot_world_lidar_mat.T # R_inv
        # --- 2. Get TFs for ROTATION (drone_world -> camera_0) ---
        
        # Q_world_lidar: Quaternion for drone_world -> lidar_link
        q_world_lidar = np.array([
            self.static_transform.rotation.x,
            self.static_transform.rotation.y,
            self.static_transform.rotation.z,
            self.static_transform.rotation.w
        ])
        
        # Q_cam_lidar: We defined this globally
        
        # Q_cam_world = Q(lidar->cam) * Q(world->lidar)
        # This is the total rotation from the drone's *static* world frame 
        # to the camera frame
        q_cam_world = tf_transformations.quaternion_multiply(Q_CAM_LIDAR, q_world_lidar)
        
        with open(self.timestamp_file, 'r') as ts_file:
            reader = csv.reader(ts_file)
            next(reader) # Skip header
            
            for row in reader:
                frame_id = int(row[0])
                frame_id_str = f"{frame_id:06d}"
                timestamp_ns = int(row[1])
                
                self.write_calib_file(frame_id_str)
                marker_msg = self.find_closest_bbox(timestamp_ns)
                
                if not marker_msg:
                    self.get_logger().warn(f"No bbox message for frame {frame_id}. Skipping.")
                    open(os.path.join(self.labels_dir, f"{frame_id_str}.txt"), 'w').close()
                    continue
                
                # --- Step 1: Get data in drone_world frame ---
                pos_world = np.array([
                    marker_msg.pose.position.x,
                    marker_msg.pose.position.y,
                    marker_msg.pose.position.z
                ])
                q_marker_world = np.array([ # Drone's orientation in world
                    marker_msg.pose.orientation.x,
                    marker_msg.pose.orientation.y,
                    marker_msg.pose.orientation.z,
                    marker_msg.pose.orientation.w
                ])
                
                # --- Step 2: Transform POSITION to lidar frame, then apply global transform ---
                # P_lidar = R_inv * (P_world - t)
                pos_lidar = np.dot(tf_rot_lidar_world_mat, (pos_world - tf_trans_world_lidar))
                
                # Apply the same global transformation as the point cloud
                pos_lidar_transformed = np.dot(R_GLOBAL, pos_lidar)
                
                # P_cam = R_cam_lidar * P_lidar_transformed + T_cam_lidar
                pos_cam_center = np.dot(R_CAM_LIDAR, pos_lidar_transformed) + T_CAM_LIDAR
                
                # --- Step 3: Transform ORIENTATION to camera_0 frame ---
                # Q_final_cam = Q(world->cam) * Q(marker->world)
                q_final_cam = tf_transformations.quaternion_multiply(q_cam_world, q_marker_world)
                
                # --- START: ROTATION LOGIC WITH GLOBAL TRANSFORM ---
                
                # Convert final orientation to a 3x3 rotation matrix
                R_final_cam = tf_transformations.quaternion_matrix(q_final_cam)[:3, :3]
                
                # Apply the global transform to the rotation matrix
                # This ensures the orientation is consistent with the transformed position
                R_final_cam_transformed = np.dot(R_CAM_LIDAR, np.dot(R_GLOBAL, np.dot(R_CAM_LIDAR.T, R_final_cam)))
                
                # Define the drone's "forward" vector (in its own body frame)
                v_body_fwd = np.array([1, 0, 0])
                
                # Rotate this vector into the camera frame
                v_cam_fwd = R_final_cam_transformed.dot(v_body_fwd)
                
                # ry_kitti is the yaw angle of this forward vector in the
                # camera's X-Z "ground plane".
                # We use atan2(x, z)
                ry_kitti = np.arctan2(v_cam_fwd[0], v_cam_fwd[2])
                
                # --- Step 4: Transform dimensions ---
                h_kitti = marker_msg.scale.z # h = z-dim
                w_kitti = marker_msg.scale.y # w = y-dim
                l_kitti = marker_msg.scale.x # l = x-dim
                
                # --- Step 5: Format for KITTI label file ---
                # Get (x, y, z) in camera 0 frame (bottom-center)
                x_kitti = pos_cam_center[0]
                y_kitti = pos_cam_center[1] + (h_kitti / 2.0)
                z_kitti = pos_cam_center[2]
                
                # Calculate alpha (observation angle)
                # alpha = ry - atan2(x_center, z_center)
                alpha_kitti = ry_kitti - np.arctan2(x_kitti, z_kitti)
                # Wrap alpha to be [-pi, pi]
                alpha_kitti = (alpha_kitti + np.pi) % (2 * np.pi) - np.pi
                
                # --- END: ROTATION LOGIC ---
                
                object_type = "Drone"
                truncation = 0.0
                occlusion = 0
                bbox_2d = [0.0, 0.0, 0.0, 0.0]
                
                kitti_line = (
                    f"{object_type} {truncation:.2f} {occlusion} {alpha_kitti:.2f} "
                    f"{bbox_2d[0]:.2f} {bbox_2d[1]:.2f} {bbox_2d[2]:.2f} {bbox_2d[3]:.2f} "
                    f"{h_kitti:.2f} {w_kitti:.2f} {l_kitti:.2f} "
                    f"{x_kitti:.2f} {y_kitti:.2f} {z_kitti:.2f} {ry_kitti:.2f}"
                )
                
                output_filename = os.path.join(self.labels_dir, f"{frame_id_str}.txt")
                with open(output_filename, 'w') as f:
                    f.write(kitti_line + '\n')
                
                if frame_id % 10 == 0:
                    self.get_logger().info(f"Processed label & calib for frame {frame_id}")
                    
        self.get_logger().info("Finished processing all labels and calibration files.")
    
    def run(self):
        type_map = self.get_topic_map()
        if self.bbox_topic not in type_map or self.tf_topic not in type_map:
            self.get_logger().error("Required topics not in bag.")
            return
        self.find_static_transform(type_map)
        self.cache_bbox_messages(type_map)
        self.process_labels()

def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Convert ROS bag Marker to KITTI label .txt files.')
    
    parser.add_argument(
        '--bag-path', 
        type=str, 
        required=True, 
        help='Path to the ROS bag file (e.g., /path/to/bag.db3)')
    
    parser.add_argument(
        '--output-path', 
        type=str, 
        required=True, 
        help='Path to the output KITTI directory')
    
    parser.add_argument(
        '--bbox-topic', 
        type=str, 
        default='/drone_bbox', 
        help='The Marker topic name')
    
    parser.add_argument(
        '--tf-topic', 
        type=str, 
        default='/tf_static', 
        help='The TF static topic name')
    
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