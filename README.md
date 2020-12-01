# ROS Object Detection Package

This repo includes a ros_object_detection package that calls a maskrcnn inference service to run bounding box based object detection and semantic image segmentation. It considers two object classes only: person and the background.

Subscribes to the topics:

1. ```/camera/depth_registered/points``` : contains colored point clouds published in the color optical frame of a RealSense D435 depth camera. Color images used for object detection are extracted from their reference colored point clouds.


Publishes into the topics:

1. ```/object_detection/detected_objects``` : contains color images enhanced by segmentation masks and bounding boxes.
2. ```/object_detection/results```   : contains Mask-RCNN person instance segmentation masks, and bounding boxes in pixels.
3. ```/object_detection/pc2```       : contains the original colored point cloud, published in the color optical frame. 
4. ```/object_detection/image_raw``` : contains the original RGB image extracted from the reference colored point cloud.


# Installation

Requires:

1. Ubuntu 18.4
2. ROS Melodic
3. Nvidia GPU
4. Python 2.7

Installing dependencies with pip is recommended. A requirements.txt file is included in the project folder. To install the required dependencies inside a virtual environment run:

```
python -m virtualenv <name_of_the_virtualenv>
source <name_of_the_virtualenv>/bin/activate
pip install -r requirements.txt
```


Inference was done on an Nvidia GeForce GTX 1660 Ti.

## Usage

For testing the ros_object_detection package, a patched version of the RealSense ROS wrapper (available under https://github.com/timeamagyar/realsense-ros/tree/ldrs_integration) must be installed on the host system. Testing requires a sample rosbag recorded with a RealSense D435 camera.

```
# cd into catkin ws
cd ~/catkin_ws
# in order to recognize packages run
source ~/catkin_ws/devel/setup.bash
# activate the python virtual environment
source  ~/<name_of_the_virtualenv>/bin/activate 
# check that the object detection package is registered in the local ros system
rospack list | grep object_detection
# check that the realsense row wrapper package is registered in the local ros system
rospack list | grep realsense2_camera
# start roscore
roscore
# cd into realsense2_camera package folder, and replay sample RealSense D435 bag with the realsense-ros wrapper to publish colored point clouds into the topic `/camera/depth_registered/points`
roslaunch realsense2_camera rs_from_file.launch rosbag_filename:=<path_to_realsense_raw_bag>
# cd into ros_object_detection package folder, and start object detection node
roslaunch object_detection object_detection.launch
# start rviz for visualization
rosrun rviz rviz
# to list available topics run
rostopic list
```

After launching the object detection package, 2D bounding box and segmentation results can be recorded into a ROS bag, needed by LDLS (3D semantic segmentation algorithm). For this run:

```
rosbag record --buffsize=0 /object_detection/detected_objects /object_detection/results /object_detection/pc2 /object_detection/image_raw --output-name=<file_name>.bag

```

Example: 

```

rosbag record --buffsize=0 /object_detection/detected_objects /object_detection/results /object_detection/pc2 /object_detection/image_raw --output-name=rs_20191213_112223.bag

```

To record camera intrinsics, needed by LDLS (3D semantic segmentation algorithm) as well run:

```
rosbag play <realsense_raw_bag>
rosbag record --buffsize=0 /device_0/sensor_1/Color_0/info/camera_info --output-name=<file_name>_intrinsics.bag

```

Example: 

```
rosbag play <realsense_raw_bag>
rosbag record --buffsize=0 /device_0/sensor_1/Color_0/info/camera_info --output-name=rs_20191213_112223_intrinsics.bag

```

## Troubleshooting

If you encounter the following error 'ImportError: No module named rospkg' simply run:

```export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages```

Make sure the following is on your python path:

```
# running echo $PYTHONPATH should return
/<path_to_catkin_ws>/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages:/usr/lib/python2.7/dist-packages

```


