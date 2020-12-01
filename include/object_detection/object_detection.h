#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "object_detection/Mask_RCNNResponse.h"
#include "object_detection/Mask_RCNN.h"
#include "object_detection/Result.h"
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud_conversion.h>


namespace object_detection {
    class ObjectDetect {
    public:
        explicit ObjectDetect(ros::NodeHandle &nh);

        virtual ~ObjectDetect();

        void registerServiceClient();

        void registerPublisher();

        void registerSubscriber();

        /**
         *@brief Callback for splitting incoming pc2 for segmentation
         *
         *@param pc2 aligned to color frame
         */
        void cbSplit(const sensor_msgs::PointCloud2::ConstPtr &points);


    private:

        /**
          *@brief Node handle for object detection.
          */
        ros::NodeHandle nh_;

        /**
          *@brief Service client for object detection service.
          */
        ros::ServiceClient detect_client_;

        /**
          *@brief ROS Image transport.
          */
        image_transport::ImageTransport it_;

        /**
          *@brief Image publisher that publishes image with bounding boxes.
          */
        image_transport::Publisher detected_objects_pub_;

        /**
         *@brief ROS result msg publisher
         */
        ros::Publisher result_pub_;

        /**
        *@brief ROS registered pc2 publisher after split
        */
        ros::Publisher pc2_pub_;

        ros::Subscriber sub_pc_2_;

        /**
         * @brief ROS rgb image publisher after split
         */
        image_transport::Publisher image_pub_;

    };
}
#endif //OBJECT_DETECTION_H

