cmake_minimum_required(VERSION 2.8.3)
project(object_detection)

add_definitions(-std=c++11)

# non catkin packages
find_package(PCL REQUIRED)

# catkin packages specified as components
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Result.msg
)

add_service_files(DIRECTORY srv
  FILES
  Mask_RCNN.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
)

# catkin_package is a cmake macro that needs to be called
# before declaring any other targets with add_executable() or add_library()
# INCLUDE_DIRS: the exported include paths for the custom catkin package
# LIBRARIES: the exported libraries from the project
# CATKIN_DEPENDS: other catkin projects that this project is depending on
# DEPENDS: non catkin based projects that this project depends on
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES object_detection
  CATKIN_DEPENDS
    cv_bridge
    image_transport 
    roscpp 
    rospy
    sensor_msgs
    std_msgs
    message_runtime
)

# prior to specifying targets, it needs to be specified where resources can be found for said targets
# specifically header files and libraries
# include paths: include_directories(): where can header files be found for the targets being built
# library paths: link_directories(): where are libraries located that executable targets build against - this is not recommended though
# all catkin and cmake packages automatically have their link information added when they are find_packaged !!!
include_directories(
  include 
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# library target - they are usually used by executable targets
add_library(${PROJECT_NAME}
   src/object_detection.cpp
)
add_definitions(${PCL_DEFINITIONS})

target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
)

# executable target - programs we can run
add_executable(${PROJECT_NAME}_node
  src/object_detection_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   object_detection
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest_gtest(test_object_detection
    test/test_object_detection_launch.test
    src/test/test_object_detection.cpp
  )
  target_link_libraries(test_object_detection
    ${catkin_LIBRARIES}
    object_detection
  )
endif()


