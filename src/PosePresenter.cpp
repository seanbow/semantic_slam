#include "semantic_slam/PosePresenter.h"

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Quaternion.h>

void PosePresenter::setup()
{
    pub_pose_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose", 10);
}

void PosePresenter::present()
{
    // Publish the last (most recent) pose
    NodeInfoPtr node = graph_->findLastNode('x');

    if (!node) return;

    gtsam::Pose3 pose;
    bool got_pose = graph_->getEstimate(node->symbol(), pose);

    if (!got_pose) return;

    Eigen::MatrixXd cov;
    bool got_cov = graph_->marginalCovariance(node->symbol(), cov);

    if (!got_cov) return;

    gtsam::Quaternion q = pose.rotation().toQuaternion();
    gtsam::Point3 p = pose.translation();

    double Z_SCALE = 1.0;

    geometry_msgs::PoseWithCovarianceStampedPtr msg_pose(new geometry_msgs::PoseWithCovarianceStamped);
    msg_pose->header.frame_id = "/map";
    msg_pose->header.stamp = *node->time();
    msg_pose->pose.pose.position.x = p.x();
    msg_pose->pose.pose.position.y = p.y();
    msg_pose->pose.pose.position.z = Z_SCALE * p.z();
    msg_pose->pose.pose.orientation.x = q.x();
    msg_pose->pose.pose.orientation.y = q.y();
    msg_pose->pose.pose.orientation.z = q.z();
    msg_pose->pose.pose.orientation.w = q.w();

    // note: ROS expects covariance of the form 
    // [ Ppp Ppq ] 
    // [ Pqp Pqq ]
    //
    // but GTSAM provides it as
    // [ Pqq Pqp ]
    // [ Ppq Ppp ]
    //
    // Make a new matrix in the right format
    Eigen::Matrix<double, 6, 6> P2;

    P2.block<3,3>(0,0) = cov.block<3,3>(3,3);
    P2.block<3,3>(0,3) = cov.block<3,3>(3,0);
    P2.block<3,3>(3,0) = cov.block<3,3>(0,3);
    P2.block<3,3>(3,3) = cov.block<3,3>(0,0);

    // ROS_INFO_STREAM("Covariance: \n" << P2);


    eigenToBoostArray<6,6>(P2, msg_pose->pose.covariance);

    pub_pose_.publish(msg_pose);

}
