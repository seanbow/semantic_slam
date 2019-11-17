#include <ros/ros.h>

#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/PosePresenter.h"
#include "semantic_slam/TrajectoryPresenter.h"
#include "semantic_slam/ObjectMeshPresenter.h"


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "semslam");
    
    // FactorGraphSupervisor supervisor;

    auto odom_handler = util::allocate_aligned<OdometryHandler>();

    auto mapper = util::allocate_aligned<SemanticMapper>();
    mapper->setOdometryHandler(odom_handler);

    // Start message handling thread
    ros::AsyncSpinner message_spinner(1);
    message_spinner.start();
    
    // Setup and add presenters
    auto pose_presenter = util::allocate_aligned<PosePresenter>();
    auto trajectory_presenter = util::allocate_aligned<TrajectoryPresenter>();
    auto object_presenter = util::allocate_aligned<ObjectMeshPresenter>();

    mapper->addPresenter(pose_presenter);
    mapper->addPresenter(trajectory_presenter);
    mapper->addPresenter(object_presenter);

    mapper->start();

}