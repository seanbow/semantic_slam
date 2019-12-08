#include <ros/ros.h>

#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/ExternalOdometryHandler.h"
#include "semantic_slam/PosePresenter.h"
#include "semantic_slam/TrajectoryPresenter.h"
#include "semantic_slam/ObjectMeshPresenter.h"
#include "semantic_slam/GeometricFeatureHandler.h"
#include "semantic_slam/GeometricMapPresenter.h"
#include "semantic_slam/ObjectKeypointPresenter.h"
#include "semantic_slam/SemanticCovisibilityPresenter.h"
#include "semantic_slam/GeometricCovisibilityPresenter.h"


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "semslam");
    
    auto mapper = util::allocate_aligned<SemanticMapper>();

    auto odom_handler = util::allocate_aligned<ExternalOdometryHandler>();
    mapper->setOdometryHandler(odom_handler);

    auto geom_handler = util::allocate_aligned<GeometricFeatureHandler>();
    mapper->setGeometricFeatureHandler(geom_handler);
    
    // Setup and add presenters
    // auto pose_presenter = util::allocate_aligned<PosePresenter>();
    // auto trajectory_presenter = util::allocate_aligned<TrajectoryPresenter>();
    // auto object_presenter = util::allocate_aligned<ObjectMeshPresenter>();
    // auto geom_presenter = util::allocate_aligned<GeometricMapPresenter>();
    // auto kp_presenter = util::allocate_aligned<ObjectKeypointPresenter>();

    mapper->addPresenter(util::allocate_aligned<PosePresenter>());
    mapper->addPresenter(util::allocate_aligned<TrajectoryPresenter>());
    mapper->addPresenter(util::allocate_aligned<ObjectMeshPresenter>());
    mapper->addPresenter(util::allocate_aligned<GeometricMapPresenter>());
    mapper->addPresenter(util::allocate_aligned<ObjectKeypointPresenter>());
    mapper->addPresenter(util::allocate_aligned<SemanticCovisibilityPresenter>());
    mapper->addPresenter(util::allocate_aligned<GeometricCovisibilityPresenter>());

    // Start message handling thread
    ros::AsyncSpinner message_spinner(1);
    message_spinner.start();

    mapper->start();

}