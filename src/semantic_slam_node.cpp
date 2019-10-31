#include <ros/ros.h>

#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/Handler.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "semslam");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    FactorGraph graph;

    std::vector<Handler> handlers;
}