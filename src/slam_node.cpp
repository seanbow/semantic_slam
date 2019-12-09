#include <ros/ros.h>

int
main()
{
    ros::init(argc, argv, "semslam");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
}