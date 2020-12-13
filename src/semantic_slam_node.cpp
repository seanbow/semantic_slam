#include <ros/ros.h>

#include "semantic_slam/ExternalOdometryHandler.h"
#include "semantic_slam/GeometricFeatureHandler.h"
#include "semantic_slam/InertialOdometryHandler.h"
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/presenters/GeometricCovisibilityPresenter.h"
#include "semantic_slam/presenters/GeometricMapPresenter.h"
#include "semantic_slam/presenters/ObjectKeypointPresenter.h"
#include "semantic_slam/presenters/ObjectMeshPresenter.h"
#include "semantic_slam/presenters/ObjectPosePresenter.h"
#include "semantic_slam/presenters/OdometryTransformPresenter.h"
#include "semantic_slam/presenters/PoseTransformPresenter.h"
#include "semantic_slam/presenters/PosePresenter.h"
#include "semantic_slam/presenters/SemanticCovisibilityPresenter.h"
#include "semantic_slam/presenters/TrajectoryPresenter.h"

#include <signal.h>
#include <glog/logging.h>

int
main(int argc, char* argv[])
{
    ros::init(argc, argv, "semslam");

    ros::NodeHandle pnh("~");

    std::string odometry_type;
    if (!pnh.getParam("odometry_type", odometry_type)) {
        ROS_ERROR("Unable to read odometry_type parameter!");
        return 1;
    }

    boost::shared_ptr<OdometryHandler> odom_handler;

    if (odometry_type == "external") {
        odom_handler = util::allocate_aligned<ExternalOdometryHandler>();
    } else if (odometry_type == "inertial") {
        odom_handler = util::allocate_aligned<InertialOdometryHandler>();
    } else {
        ROS_ERROR_STREAM("Unknown odometry type: " << odometry_type);
        return 1;
    }

    auto mapper = util::allocate_aligned<SemanticMapper>();
    auto geom_handler = util::allocate_aligned<GeometricFeatureHandler>();

    // Setup and add presenters
    mapper->setOdometryHandler(odom_handler);
    mapper->setGeometricFeatureHandler(geom_handler);

    mapper->addPresenter(util::allocate_aligned<PosePresenter>());
    mapper->addPresenter(util::allocate_aligned<TrajectoryPresenter>());
    mapper->addPresenter(util::allocate_aligned<ObjectMeshPresenter>());
    mapper->addPresenter(util::allocate_aligned<ObjectPosePresenter>());
    mapper->addPresenter(util::allocate_aligned<GeometricMapPresenter>());
    mapper->addPresenter(util::allocate_aligned<ObjectKeypointPresenter>());
    mapper->addPresenter(util::allocate_aligned<OdometryTransformPresenter>());
    mapper->addPresenter(util::allocate_aligned<PoseTransformPresenter>());
    mapper->addPresenter(
      util::allocate_aligned<SemanticCovisibilityPresenter>());
    mapper->addPresenter(
      util::allocate_aligned<GeometricCovisibilityPresenter>());

    google::InitGoogleLogging(argv[0]);

    // Start message handling thread
    ros::AsyncSpinner message_spinner(1);
    message_spinner.start();

    mapper->start();
}