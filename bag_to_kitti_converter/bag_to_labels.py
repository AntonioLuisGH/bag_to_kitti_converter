#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from visualization_msgs.msg import Marker
import os
import argparse
import csv
import numpy as np
import tf_transformations
import bisect
import sys

# --- HELPER FUNCTIONS ---
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
    ])

# --- KITTI CONSTANTS ---
R_CAM_LIDAR = np.array([[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]])
T_CAM_LIDAR = np.array([0, 0, 0])
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
        
        self.labels_dir = os.path.join(self.output_path, 'label_2')
        self.calib_dir = os.path.join(self.output_path, 'calib')
        self.timestamp_file = os.path.join(self.output_path, 'timestamps.csv')
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.calib_dir, exist_ok=True)
        
        # Store ALL transforms here: [(timestamp, translation, rotation_quat, R_GLOBAL_MAT), ...]
        self.tf_cache = [] 
        self.bbox_messages = []
        
        # Target frames
        self.parent_frame = 'drone_world'
        self.child_frame = 'lidar_link'

    def get_bag_reader(self):
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)
        return reader

    def get_topic_map(self):
        reader = self.get_bag_reader()
        type_map = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        del reader
        return type_map

    def cache_all_transforms(self, type_map):
        """
        Reads EVERY transform message in the bag and stores it sorted by time.
        This allows us to handle moving TFs.
        """
        self.get_logger().info(f"Caching DYNAMIC transforms from: {self.tf_topic}...")
        reader = self.get_bag_reader()
        storage_filter = rosbag2_py.StorageFilter(topics=[self.tf_topic])
        reader.set_filter(storage_filter)
        
        msg_type = get_message(type_map[self.tf_topic])
        
        count = 0
        while reader.has_next():
            (topic, data, timestamp_ns) = reader.read_next()
            if topic == self.tf_topic:
                tf_msg = deserialize_message(data, msg_type)
                for t in tf_msg.transforms:
                    if t.header.frame_id == self.parent_frame and t.child_frame_id == self.child_frame:
                        
                        # 1. Extract Translation
                        trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                        
                        # 2. Extract Rotation (Quaternion)
                        q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
                        
                        # 3. Compute the R_GLOBAL matrix specifically for this moment in time
                        # (As per your requirement to use the TF rotation as the global fix)
                        r_global = quaternion_to_rotation_matrix(q)
                        
                        # Store tuple: (Time, Translation, Quaternion, R_Global_Matrix)
                        # We use the message header stamp, not the bag timestamp, for better precision
                        msg_time_ns = t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec
                        self.tf_cache.append((msg_time_ns, trans, np.array(q), r_global))
                        count += 1

        # Sort by timestamp to enable binary search
        self.tf_cache.sort(key=lambda x: x[0])
        self.get_logger().info(f"Cached {count} transforms.")
        del reader

    def get_interpolated_transform(self, query_time_ns):
        """
        Finds the transform closest in time to the query_time_ns.
        """
        if not self.tf_cache:
            return None, None, None
            
        # Extract just the timestamps for bisect
        timestamps = [x[0] for x in self.tf_cache]
        idx = bisect.bisect_left(timestamps, query_time_ns)
        
        # Find nearest neighbor
        if idx == 0:
            best_idx = 0
        elif idx == len(timestamps):
            best_idx = -1
        else:
            before = timestamps[idx - 1]
            after = timestamps[idx]
            if (query_time_ns - before) < (after - query_time_ns):
                best_idx = idx - 1
            else:
                best_idx = idx
                
        return self.tf_cache[best_idx][1], self.tf_cache[best_idx][2], self.tf_cache[best_idx][3]

    def cache_bbox_messages(self, type_map):
        # Same as before...
        self.get_logger().info(f"Caching bbox messages...")
        reader = self.get_bag_reader()
        reader.set_filter(rosbag2_py.StorageFilter(topics=[self.bbox_topic]))
        msg_type = get_message(type_map[self.bbox_topic])
        while reader.has_next():
            (topic, data, timestamp_ns) = reader.read_next()
            bbox_msg = deserialize_message(data, msg_type)
            self.bbox_messages.append((timestamp_ns, bbox_msg))
        self.bbox_messages.sort(key=lambda x: x[0])
        del reader

    def find_closest_bbox(self, target_timestamp_ns):
        # Same as before...
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
        # Same as before...
        P_identity = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
        R0_rect = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00"
        Tr_imu = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
        tr_data = np.hstack((R_CAM_LIDAR, T_CAM_LIDAR.reshape(3, 1)))
        tr_str = " ".join([f"{x:.6e}" for x in tr_data.flatten()])
        content = f"P0: {P_identity}\nP1: {P_identity}\nP2: {P_identity}\nP3: {P_identity}\nR0_rect: {R0_rect}\nTr_velo_to_cam: {tr_str}\nTr_imu_to_velo: {Tr_imu}\n"
        with open(os.path.join(self.calib_dir, f"{frame_id_str}.txt"), 'w') as f: f.write(content)

    def process_labels(self):
        if not os.path.exists(self.timestamp_file): return
        if not self.tf_cache: 
            self.get_logger().error("No transforms found!")
            return

        with open(self.timestamp_file, 'r') as ts_file:
            reader = csv.reader(ts_file)
            next(reader) # Skip header
            
            for row in reader:
                frame_id = int(row[0])
                frame_id_str = f"{frame_id:06d}"
                timestamp_ns = int(row[1])
                
                self.write_calib_file(frame_id_str)
                marker_msg = self.find_closest_bbox(timestamp_ns)
                
                # --- KEY CHANGE: Get TF specifically for THIS timestamp ---
                tf_trans, tf_quat, tf_R_GLOBAL = self.get_interpolated_transform(timestamp_ns)
                
                if marker_msg is None or tf_trans is None:
                    open(os.path.join(self.labels_dir, f"{frame_id_str}.txt"), 'w').close()
                    continue
                
                # --- Calculate Matrices for THIS frame ---
                # World -> Lidar
                tf_rot_world_lidar_mat = tf_transformations.quaternion_matrix(tf_quat)[:3, :3]
                tf_rot_lidar_world_mat = tf_rot_world_lidar_mat.T # Inverse

                # Rotation quats
                q_world_lidar = tf_quat
                q_cam_world = tf_transformations.quaternion_multiply(Q_CAM_LIDAR, q_world_lidar)

                # --- Step 1: Marker in World ---
                pos_world = np.array([marker_msg.pose.position.x, marker_msg.pose.position.y, marker_msg.pose.position.z])
                q_marker_world = np.array([marker_msg.pose.orientation.x, marker_msg.pose.orientation.y, marker_msg.pose.orientation.z, marker_msg.pose.orientation.w])

                # --- Step 2: Transform Position (Dynamic) ---
                # Move to Lidar frame using the DYNAMIC transform
                pos_lidar = np.dot(tf_rot_lidar_world_mat, (pos_world - tf_trans))
                
                # Apply the DYNAMIC global fix (based on the current TF)
                pos_lidar_transformed = np.dot(tf_R_GLOBAL, pos_lidar)
                
                # Move to Camera
                pos_cam_center = np.dot(R_CAM_LIDAR, pos_lidar_transformed) + T_CAM_LIDAR

                # --- Step 3: Transform Orientation ---
                q_final_cam = tf_transformations.quaternion_multiply(q_cam_world, q_marker_world)
                R_final_cam = tf_transformations.quaternion_matrix(q_final_cam)[:3, :3]
                
                # Apply Dynamic Global Fix to rotation
                R_final_cam_transformed = np.dot(R_CAM_LIDAR, np.dot(tf_R_GLOBAL, np.dot(R_CAM_LIDAR.T, R_final_cam)))
                
                # Yaw calc
                v_cam_fwd = R_final_cam_transformed.dot(np.array([1, 0, 0]))
                ry_kitti = np.arctan2(v_cam_fwd[0], v_cam_fwd[2])

                # --- Step 4: Output ---
                h, w, l = marker_msg.scale.z, marker_msg.scale.y, marker_msg.scale.x
                x_kitti, y_kitti, z_kitti = pos_cam_center[0], pos_cam_center[1] + (h/2.0), pos_cam_center[2]
                
                alpha = ry_kitti - np.arctan2(x_kitti, z_kitti)
                alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

                kitti_line = f"Drone 0.00 0 {alpha:.2f} 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {x_kitti:.2f} {y_kitti:.2f} {z_kitti:.2f} {ry_kitti:.2f}"
                
                with open(os.path.join(self.labels_dir, f"{frame_id_str}.txt"), 'w') as f:
                    f.write(kitti_line + '\n')
                    
                if frame_id % 10 == 0: self.get_logger().info(f"Processed {frame_id}")

    def run(self):
        type_map = self.get_topic_map()
        if self.tf_topic not in type_map:
             self.get_logger().error("TF Topic not found")
             return
        self.cache_all_transforms(type_map)
        self.cache_bbox_messages(type_map)
        self.process_labels()

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--bbox-topic', type=str, default='/drone_bbox')
    parser.add_argument('--tf-topic', type=str, default='/tf_static') # Change to /tf if needed
    parsed_args = parser.parse_args(args=rclpy.utilities.remove_ros_args(args=sys.argv[1:]))
    node = BagToLabels(parsed_args.bag_path, parsed_args.output_path, parsed_args.bbox_topic, parsed_args.tf_topic)
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()