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
        
        self.static_transform = None 
        self.bbox_messages = [] 
        
        # Default to Identity (will be overwritten by find_static_transform)
        self.R_GLOBAL = np.eye(3) 
    
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
        
        if self.tf_topic not in type_map:
             self.get_logger().error(f"TF topic {self.tf_topic} not found!")
             return

        msg_type = get_message(type_map[self.tf_topic])
        found = False

        while reader.has_next():
            (topic, data, _) = reader.read_next()
            if topic == self.tf_topic:
                tf_msg = deserialize_message(data, msg_type)
                for transform in tf_msg.transforms:
                    if transform.header.frame_id == 'drone_world' and transform.child_frame_id == 'lidar_link':
                        self.static_transform = transform.transform
                        
                        # --- NEW: EXTRACT R_GLOBAL FROM TF ---
                        qx = transform.transform.rotation.x
                        qy = transform.transform.rotation.y
                        qz = transform.transform.rotation.z
                        qw = transform.transform.rotation.w
                        
                        # Calculate the Rotation Matrix exactly as the Pointcloud script did
                        self.R_GLOBAL = quaternion_to_rotation_matrix([qx, qy, qz, qw])
                        
                        self.get_logger().info(f"Found static transform & Updated R_GLOBAL.")
                        found = True
                        break
            if found: break
            
        del reader
        if not found:
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
        self.get_logger().info("Processing labels and calibration files...")
        
        if not os.path.exists(self.timestamp_file):
            self.get_logger().error(f"Timestamp file not found: {self.timestamp_file}")
            return
        if not self.static_transform:
            self.get_logger().error("Static transform not found. Cannot process labels.")
            return
        
        # --- 1. Calculate Transformation Matrices ---
        # Translation from World to Lidar
        tf_trans_world_lidar = np.array([
            self.static_transform.translation.x,
            self.static_transform.translation.y,
            self.static_transform.translation.z
        ])
        
        # Rotation from World to Lidar
        tf_rot_world_lidar_mat = tf_transformations.quaternion_matrix([
            self.static_transform.rotation.x,
            self.static_transform.rotation.y,
            self.static_transform.rotation.z,
            self.static_transform.rotation.w
        ])[:3, :3]
        
        # Inverse Rotation (Lidar to World orientation)
        tf_rot_lidar_world_mat = tf_rot_world_lidar_mat.T 

        # Q_world_lidar: Quaternion for drone_world -> lidar_link
        q_world_lidar = np.array([
            self.static_transform.rotation.x,
            self.static_transform.rotation.y,
            self.static_transform.rotation.z,
            self.static_transform.rotation.w
        ])
        
        # Total rotation from drone static frame to camera frame
        q_cam_world = tf_transformations.quaternion_multiply(Q_CAM_LIDAR, q_world_lidar)
        
        with open(self.timestamp_file, 'r') as ts_file:
            reader = csv.reader(ts_file)
            next(reader) 
            
            for row in reader:
                frame_id = int(row[0])
                frame_id_str = f"{frame_id:06d}"
                timestamp_ns = int(row[1])
                
                self.write_calib_file(frame_id_str)
                marker_msg = self.find_closest_bbox(timestamp_ns)
                
                if not marker_msg:
                    open(os.path.join(self.labels_dir, f"{frame_id_str}.txt"), 'w').close()
                    continue
                
                # --- Step 1: Get raw marker pose in drone_world frame ---
                pos_world = np.array([
                    marker_msg.pose.position.x,
                    marker_msg.pose.position.y,
                    marker_msg.pose.position.z
                ])
                
                q_marker_world = np.array([ 
                    marker_msg.pose.orientation.x,
                    marker_msg.pose.orientation.y,
                    marker_msg.pose.orientation.z,
                    marker_msg.pose.orientation.w
                ])
                
                # --- Step 2: Transform POSITION ---
                # A. Move from World Frame to Lidar Frame (Standard ROS TF)
                # P_lidar = R_inv * (P_world - T)
                pos_lidar = np.dot(tf_rot_lidar_world_mat, (pos_world - tf_trans_world_lidar))
                
                # B. Apply the "Global Stabilization" Transform
                # This must match the pointcloud script exactly!
                # P_stabilized = R_GLOBAL * P_lidar
                pos_lidar_transformed = np.dot(self.R_GLOBAL, pos_lidar)
                
                # C. Move to Camera Frame
                # P_cam = R_cam_lidar * P_stabilized + T_cam_lidar
                pos_cam_center = np.dot(R_CAM_LIDAR, pos_lidar_transformed) + T_CAM_LIDAR
                
                # --- Step 3: Transform ORIENTATION ---
                # Q_final_cam = Q(world->cam) * Q(marker->world)
                q_final_cam = tf_transformations.quaternion_multiply(q_cam_world, q_marker_world)
                
                # Convert to Matrix
                R_final_cam = tf_transformations.quaternion_matrix(q_final_cam)[:3, :3]
                
                # Apply the same global transform logic to the rotation
                # R_final = R_CAM_LIDAR * R_GLOBAL * R_CAM_LIDAR_INV * R_current
                R_final_cam_transformed = np.dot(R_CAM_LIDAR, np.dot(self.R_GLOBAL, np.dot(R_CAM_LIDAR.T, R_final_cam)))
                
                # Calculate Ry (Yaw)
                v_body_fwd = np.array([1, 0, 0])
                v_cam_fwd = R_final_cam_transformed.dot(v_body_fwd)
                ry_kitti = np.arctan2(v_cam_fwd[0], v_cam_fwd[2])
                
                # --- Step 4: Dimensions & Format ---
                h_kitti = marker_msg.scale.z 
                w_kitti = marker_msg.scale.y 
                l_kitti = marker_msg.scale.x 
                
                # KITTI Y is at the bottom of the object, Marker Y is center.
                # Since Camera Y is DOWN, "Bottom" has a HIGHER Y value.
                x_kitti = pos_cam_center[0]
                y_kitti = pos_cam_center[1] + (h_kitti / 2.0) 
                z_kitti = pos_cam_center[2]
                
                alpha_kitti = ry_kitti - np.arctan2(x_kitti, z_kitti)
                alpha_kitti = (alpha_kitti + np.pi) % (2 * np.pi) - np.pi
                
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
                    self.get_logger().info(f"Processed label for frame {frame_id}")
                    
        self.get_logger().info("Finished processing.")
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--bbox-topic', type=str, default='/drone_bbox')
    parser.add_argument('--tf-topic', type=str, default='/tf_static')
    
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