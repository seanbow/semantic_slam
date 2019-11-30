#include <ros/ros.h>

#include "semantic_slam/SimpleObjectTracker.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "simple_object_tracker");

    SimpleObjectTracker tracker;

    ros::spin();
}