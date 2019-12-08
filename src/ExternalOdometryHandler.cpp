#include "semantic_slam/ExternalOdometryHandler.h"

// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/slam/BetweenFactor.h>
#include "semantic_slam/CeresSE3PriorFactor.h"
#include "semantic_slam/CeresBetweenFactor.h"

#include <string>


using namespace std::string_literals;

void ExternalOdometryHandler::setup()
{
    ROS_INFO("Starting odometry handler.");
    std::string odometry_topic;
    pnh_.param("odom_topic", odometry_topic, "/zed/zed_node/odom"s);

    ROS_INFO_STREAM("Subscribing to topic " << odometry_topic);

    subscriber_ = nh_.subscribe(odometry_topic, 10000, &ExternalOdometryHandler::msgCallback, this);

    received_msgs_ = 0;
    last_keyframe_index_ = 0;

    node_chr_ = 'x';

    max_node_period_ = ros::Duration(0.5); // seconds
}

void ExternalOdometryHandler::msgCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
	received_msgs_++;

	if (received_msgs_ > 1 && msg->header.seq != last_msg_seq_ + 1) {
        ROS_ERROR_STREAM("[Drone Odometry] Error: dropped relative pose message. Expected " << last_msg_seq_ + 1 << ", got " << msg->header.seq);
    }
    // ROS_INFO_STREAM("Received relpose msg, seq " << msg->header.seq << ", time " << msg->header.stamp);
    {
    	std::lock_guard<std::mutex> lock(mutex_);
    	msg_queue_.push_back(*msg);
    }
    // cv_.notify_all();

    last_msg_seq_ = msg->header.seq;

    // cv_->notify_all();
}

Pose3 ExternalOdometryHandler::msgToPose3(const nav_msgs::Odometry& msg) const
{
    auto p_msg = msg.pose.pose.position;
    auto q_msg = msg.pose.pose.orientation;

    Eigen::Vector3d p(p_msg.x, p_msg.y, p_msg.z);
    Eigen::Quaterniond q;
    q.x() = q_msg.x;
    q.y() = q_msg.y;
    q.z() = q_msg.z;
    q.w() = q_msg.w;

    Pose3 G_p(q, p);

    return G_p;
}

Eigen::MatrixXd ExternalOdometryHandler::extractOdometryCovariance(const nav_msgs::Odometry& msg) const
{
    Eigen::Matrix<double, 6, 6> cov_tmp;
    boostArrayToEigen<6,6>(msg.pose.covariance, cov_tmp);

    // Odometry message ordering is [p q] but we want it [q p]
    Eigen::MatrixXd cov(6,6);
    cov.block<3,3>(0,0) = cov_tmp.block<3,3>(3,3);
    cov.block<3,3>(0,3) = cov_tmp.block<3,3>(3,0);
    cov.block<3,3>(3,0) = cov_tmp.block<3,3>(0,3);
    cov.block<3,3>(3,3) = cov_tmp.block<3,3>(0,0);

    return cov;
}

SemanticKeyframe::Ptr ExternalOdometryHandler::findNearestKeyframe(ros::Time t)
{
    ros::Duration shortest_duration = ros::DURATION_MAX;
    SemanticKeyframe::Ptr kf = nullptr;

    for (auto& frame : keyframes_) {
        if (abs_duration(t - frame->time()) <= shortest_duration) {
            shortest_duration = abs_duration(t - frame->time());
            kf = frame;
        }
    }

    return kf;
}

bool ExternalOdometryHandler::getRelativePoseEstimate(ros::Time t1, ros::Time t2, Pose3& T12)
{
    // Assume here that t1 is not too far ahead of nodes that are already in the graph, so:
    // auto node1 = boost::static_pointer_cast<SE3Node>(graph_->findNearestNode(node_chr_, t1));
    auto kf1 = findNearestKeyframe(t1);

    if (!kf1) return false;

    Pose3 odom1 = kf1->odometry();

    // And assume now (TODO) that t2 IS too far ahead of nodes so we just have to look in the
    // message queue for its odometry information
    if (msg_queue_.size() < 2) return false;

    auto msg_it = msg_queue_.begin();
    while (msg_it->header.stamp < t2 && msg_it != msg_queue_.end()) {
        msg_it++;
    }

    if (msg_it == msg_queue_.end()) {
        return false;
    }

    nav_msgs::Odometry msg = *msg_it;

    Pose3 odom2 = msgToPose3(msg);

    T12 = odom1.inverse() * odom2;

    return true;
}

// bool ExternalOdometryHandler::getRelativePoseJacobianEstimate(ros::Time t1, ros::Time t2, Eigen::MatrixXd& H12)
// {
//     // Assume here that t1 is not too far ahead of nodes that are already in the graph, so:
//     // auto node1 = boost::static_pointer_cast<SE3Node>(graph_->findNearestNode(node_chr_, t1));
//     auto kf1 = findNearestKeyframe(t1);

//     if (!kf1) return false;

//     Eigen::MatrixXd odom_cov1 = kf1->odometry_covariance();

//     // And assume now (TODO) that t2 IS too far ahead of nodes so we just have to look in the
//     // message queue for its odometry information
//     if (msg_queue_.size() < 2) return false;

//     auto msg_it = msg_queue_.begin();
//     while (msg_it->header.stamp < t2 && msg_it != msg_queue_.end()) {
//         msg_it++;
//     }

//     if (msg_it == msg_queue_.end()) {
//         return false;
//     }

//     nav_msgs::Odometry msg = *msg_it;

//     Eigen::MatrixXd odom_cov2 = extractOdometryCovariance(msg);

//     Eigen::LLT<Eigen::MatrixXd> chol1(odom_cov1);
//     Eigen::LLT<Eigen::MatrixXd> chol2(odom_cov2);

//     Eigen::MatrixXd L1 = chol1.matrixL();

//     H12 = chol2.matrixL() * L1.inverse();

//     return true;
// }

// CeresNodePtr ExternalOdometryHandler::getSpineNode(ros::Time time)
// {
//     auto node = graph_->findFirstNodeAfterTime(node_chr_, time);

//     // TODO check that the time is close, whether we should add one before it, etc

//     if (node) {
//         return node;
//     } else {
//         return attachSpineNode(time);
//     }
// }

// CeresNodePtr ExternalOdometryHandler::attachSpineNode(ros::Time time)
SemanticKeyframe::Ptr ExternalOdometryHandler::createKeyframe(ros::Time time)
{
    // Integrate up to time and attach a node corresponding to it

    // check size() < 2 instead of !empty(). if we just checked 
    // empty, it may be that we get pop a message just before `time` and the queue
    // becomes empty, yet the next message we receive will be after `time`. can't
    // check for this case unless we have both "straddling" messages present here
    if (msg_queue_.size() < 2) return nullptr;

    Symbol keyframe_symbol(node_chr_, last_keyframe_index_ + 1);

    // if (!last_odom_node) {
    //     // No odometry "spine" exists yet -- create a first node and anchor 
    //     // it at the origin.

    //     Pose3 origin = Pose3::Identity();

    //     Symbol origin_symbol = Symbol(node_chr_, 0);
    //     node_odom_[origin_symbol.key()] = origin; // TODO is this safe to assume odom origin = actual origin

    //     last_odom_node = util::allocate_aligned<SE3Node>(origin_symbol, msg_queue_.front().header.stamp);
    //     last_odom_node->pose() = origin;

    //     graph_->addNode(last_odom_node);
    //     graph_->setNodeConstant(last_odom_node);

    //     // TODO figure out best values for origin anchoring factor noise
    //     // double sigma_p = .01;
    //     // double sigma_q = .001;
    //     // Eigen::VectorXd sigmas(6);
    //     // sigmas << sigma_q, sigma_q, sigma_q, sigma_p, sigma_p, sigma_p;
    //     // Eigen::MatrixXd anchor_cov = sigmas.array().pow(2).matrix().asDiagonal();
    //     // auto fac = util::allocate_aligned<CeresSE3PriorFactor>(last_odom_node, origin, anchor_cov);
    //     // graph_->addFactor(fac);

    //     ROS_INFO_STREAM("Added first node " << DefaultKeyFormatter(origin_symbol) 
    //                         << " to graph, origin = " << origin);
    // }

    // // Proceed with odometry integration & new node creation
    // Symbol symbol = Symbol(node_chr_, last_odom_node->index() + 1);

    nav_msgs::Odometry msg = msg_queue_.front();

    if (keyframes_.empty()) {
        // This is the first keyframe -- create the origin frame
        SemanticKeyframe::Ptr frame = originKeyframe(msg_queue_.front().header.stamp);
        return frame;
    }

    bool good_msg = false;

    {
        std::lock_guard<std::mutex> guard(mutex_);
        msg_queue_.pop_front();

        // TODO interpolation if `time` is between message times
        while (msg_queue_.size() >= 2 && msg_queue_.front().header.stamp <= time) {
            msg = msg_queue_.front();
            msg_queue_.pop_front();
        }

        if (!msg_queue_.empty() && msg_queue_.front().header.stamp > time) {
            good_msg = true;
        }
    }

    if (!good_msg) {
        // don't have odometry messages up to the requested time yet
        return nullptr;
    }


    last_time_ = msg.header.stamp;

    SemanticKeyframe::Ptr keyframe = util::allocate_aligned<SemanticKeyframe>(keyframe_symbol, msg.header.stamp);
    SemanticKeyframe::Ptr last_keyframe = keyframes_.back();
    keyframes_.push_back(keyframe);

    Pose3 G_p_now = msgToPose3(msg);
    G_p_now.rotation().normalize();
    keyframe->odometry() = G_p_now;

    keyframe->odometry_covariance() = extractOdometryCovariance(msg);

    keyframe->image_time = time;

    Pose3 relp = last_keyframe->odometry().between(G_p_now);

    // Node that here for the initial estimate, we don't want to use the current odometry estimate.
    // We just want to use the odometry *delta* and update the previous optimized estimate
    Pose3 prev_state = last_keyframe->pose();
    keyframe->pose() = prev_state * relp;
    keyframe->graph_node()->pose() = keyframe->pose();

    // TODO use more accurate relative covariance information...
    // Eigen::Matrix6d cov;
    double sigma_p = 0.05;
    double sigma_q = 0.0075;
    Eigen::VectorXd sigmas(6);
    sigmas << sigma_q, sigma_q, sigma_q, sigma_p, sigma_p, sigma_p;
    Eigen::MatrixXd cov = sigmas.array().pow(2).matrix().asDiagonal();

    auto fac = util::allocate_aligned<CeresBetweenFactor>(last_keyframe->graph_node(),
                                                            keyframe->graph_node(),
                                                            relp,
                                                            cov);

    keyframe->spine_factor() = fac;

    // graph_->addNode(node);
    // graph_->addFactor(fac);
    // graph_->setModified();

    // ROS_INFO_STREAM("Added new odometry node " << DefaultKeyFormatter(symbol));

    // ROS_INFO_STREAM("Added spine node " << DefaultKeyFormatter(symbol) << " at time = " 
    //         << *node->time() << " (requested t = " << time << ")");

    last_keyframe_index_++;

    // ROS_INFO_STREAM("Created keyframe with time " << keyframe->time() << " and image time " << keyframe->image_time);

    return keyframe;
}

SemanticKeyframe::Ptr ExternalOdometryHandler::originKeyframe(ros::Time time) {
    SemanticKeyframe::Ptr kf = util::allocate_aligned<SemanticKeyframe>(Symbol(node_chr_, 0), time);

    kf->odometry() = Pose3::Identity();
    kf->pose() = Pose3::Identity();
    kf->graph_node()->pose() = Pose3::Identity();
    keyframes_.push_back(kf);

    return kf;
}