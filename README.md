# ROS2 Bag to KITTI Dataset Conversion

## Overview

This package converts ROS2 bag files containing PointCloud2 and Marker messages into KITTI dataset format for 3D object detection pipelines like PointPillars.

## Features

- **PointCloud2 Conversion**: Extract sensor_msgs/PointCloud2 topics to KITTI Velodyne .bin files
- **Marker to Labels**: Convert visualization_msgs/Marker bounding boxes to KITTI label .txt files
- **Transform Support**: Use static transformations (tf_static) for accurate object localization
- **Timestamp Mapping**: Generate timestamps.csv for frame-to-timestamp correspondence

## Scripts

| Script | Purpose |
|--------|---------|
| `bag_to_velodyne.py` | Converts PointCloud2 to .bin files and generates timestamps |
| `bag_to_labels.py` | Converts Marker messages to KITTI labels using transforms |

## Quick Start

### Step 1: Extract Point Clouds

```bash
ros2 run <package_name> bag_to_velodyne.py \
    --bag-path /path/to/bag.db3 \
    --output-path /path/to/output \
    --topic /lidar/avia
```

### Step 2: Extract Labels

```bash
ros2 run <package_name> bag_to_labels.py \
    --bag-path /path/to/bag.db3 \
    --output-path /path/to/output \
    --bbox-topic /drone_bbox \
    --tf-topic /tf_static
```

## Requirements

- Python 3.8+
- ROS2 Galactic or newer
- Dependencies: `numpy`, `tf-transformations`

```bash
pip install numpy tf-transformations
```

## Output Structure

```
output/
├── velodyne/          # KITTI .bin point clouds
├── label_2/           # KITTI .txt labels
└── timestamps.csv     # Frame ID to timestamp mapping
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Topic not found | Verify topic names with `ros2 bag info` |
| Missing transforms | Check `/tf_static` contains drone_world → lidar_link |
| No intensity data | Default intensity values are applied |
