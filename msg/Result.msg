std_msgs/Header header

# Bounding boxes in pixels
sensor_msgs/RegionOfInterest[] boxes

# Integer class IDs for each bounding box
int32[] class_ids

# String class IDs for each bounding box
string[] class_names

# Instance masks as Image
sensor_msgs/Image[] masks
