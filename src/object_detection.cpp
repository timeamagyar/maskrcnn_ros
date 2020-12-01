#include "object_detection/object_detection.h"

namespace object_detection {

    ObjectDetect::ObjectDetect(ros::NodeHandle &nh) :
            nh_(nh),
            it_(nh_) {
        registerServiceClient();
        registerPublisher();
        registerSubscriber();
    }


    ObjectDetect::~ObjectDetect() = default;


    void ObjectDetect::registerServiceClient() {
        detect_client_ = nh_.serviceClient<object_detection::Mask_RCNN>("run_inference_maskrcnn");
    }


    void ObjectDetect::registerPublisher() {
        detected_objects_pub_ = it_.advertise("/object_detection/detected_objects", 1500);
        result_pub_ = nh_.advertise<Result>("/object_detection/results", 1500);
        pc2_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/object_detection/pc2", 1500);
        image_pub_ = it_.advertise("/object_detection/image_raw", 1500);
        std::cout << "Publisher initialized.." << std::endl;
    }


    void ObjectDetect::registerSubscriber() {
        sub_pc_2_ = nh_.subscribe("/camera/depth_registered/points", 1500, &ObjectDetect::cbSplit, this);
        std::cout << "Subscriber initialized.." << std::endl;
    }


    void ObjectDetect::cbSplit(const sensor_msgs::PointCloud2::ConstPtr& points) {
#if 0
        // save pcd file
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*points, pcl_cloud);
        std::string filePath;
        nh_.getParam("/object_detection/annotation_path", filePath);
        std::stringstream filename;
        filename << filePath << points->header.stamp.sec << "." << std::setfill('0') << std::setw(9) << points->header.stamp.nsec << ".pcd";
        std::cout << "Saving file " << filename.str() << std::endl;
        pcl::io::savePCDFile( filename.str(), pcl_cloud, false ); // ASCII format

#endif


        sensor_msgs::ImagePtr image(new sensor_msgs::Image);
        std_msgs::Header header = points->header;
        std::cout << points->header << std::endl;
        //Copy the RGB fields of a PointCloud2 msg into sensor_msgs::Image format.
        try
        {
            pcl::toROSMsg(*points, *image);
            image->header = header;
        }
        catch (const std::runtime_error& e)
        {
            ROS_ERROR_STREAM("caught exception " << e.what() << " while splitting, skip this message");
            return;
        }

        pc2_pub_.publish(points);
        std::cout << points << std::endl;
        image_pub_.publish(image);

        object_detection::Mask_RCNN srv;
        srv.request.img_req = *image;
        detect_client_.waitForExistence();

        if (detect_client_.call(srv)) {
            detected_objects_pub_.publish(srv.response.img_res);
            result_pub_.publish(srv.response.result);
        } else {
            std::cout << "Unable to reach object detection service.\n";
        }

    }
} // object_detect
