#include "semantic_slam/OdometryHandler.h"

// #include <gtsam/slam/PriorFactor.h>
// #include <gtsam/slam/BetweenFactor.h>
#include "semantic_slam/CeresSE3PriorFactor.h"
#include "semantic_slam/CeresBetweenFactor.h"

#include <string>


using namespace std::string_literals;

void OdometryHandler::setup()
{
    ROS_INFO("Starting odometry handler.");
    std::string odometry_topic;
    pnh_.param("odom_topic", odometry_topic, "/zed/zed_node/odom"s);

    ROS_INFO_STREAM("Subscribing to topic " << odometry_topic);

    subscriber_ = nh_.subscribe(odometry_topic, 1000, &OdometryHandler::msgCallback, this);

    received_msgs_ = 0;

    node_chr_ = 'x';

    max_node_period_ = ros::Duration(0.5); // seconds
}

void OdometryHandler::msgCallback(const nav_msgs::Odometry::ConstPtr& msg)
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

Pose3 OdometryHandler::msgToPose3(const nav_msgs::Odometry& msg)
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

CeresNodePtr OdometryHandler::getSpineNode(ros::Time time)
{
    auto node = graph_->findFirstNodeAfterTime(node_chr_, time);

    // TODO check that the time is close, whether we should add one before it, etc

    if (node) {
        return node;
    } else {
        return attachSpineNode(time);
    }
}

CeresNodePtr OdometryHandler::attachSpineNode(ros::Time time)
{
    // Integrate up to time and attach a node corresponding to it

    // check size() < 2 instead of !empty(). if we just checked 
    // empty, it may be that we get pop a message just before `time` and the queue
    // becomes empty, yet the next message we receive will be after `time`. can't
    // check for this case unless we have both "straddling" messages present here
    if (msg_queue_.size() < 2) return nullptr;

    SE3NodePtr last_odom_node = graph_->findLastNode<SE3Node>(node_chr_);

    if (!last_odom_node) {
        // No odometry "spine" exists yet -- create a first node and anchor 
        // it at the origin.

        Pose3 origin = Pose3::Identity();

        Symbol origin_symbol = Symbol(node_chr_, 0);
        node_odom_[origin_symbol.key()] = origin; // TODO is this safe to assume odom origin = actual origin

        last_odom_node = util::allocate_aligned<SE3Node>(origin_symbol, msg_queue_.front().header.stamp);
        last_odom_node->pose() = origin;

        graph_->addNode(last_odom_node);
        graph_->setNodeConstant(last_odom_node);

        // TODO figure out best values for origin anchoring factor noise
        // double sigma_p = .01;
        // double sigma_q = .001;
        // Eigen::VectorXd sigmas(6);
        // sigmas << sigma_q, sigma_q, sigma_q, sigma_p, sigma_p, sigma_p;
        // Eigen::MatrixXd anchor_cov = sigmas.array().pow(2).matrix().asDiagonal();
        // auto fac = util::allocate_aligned<CeresSE3PriorFactor>(last_odom_node, origin, anchor_cov);
        // graph_->addFactor(fac);

        ROS_INFO_STREAM("Added first node " << DefaultKeyFormatter(origin_symbol) 
                            << " to graph, origin = " << origin);
    }

    // Proceed with odometry integration & new node creation
    Symbol symbol = Symbol(node_chr_, last_odom_node->index() + 1);

    nav_msgs::Odometry msg = msg_queue_.front();

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
            msg = msg_queue_.front();
            msg_queue_.pop_front();
            good_msg = true;
        }
    }

    if (!good_msg) {
        // don't have odometry messages up to the requested time yet
        return nullptr;
    }


    last_time_ = msg.header.stamp;

    Pose3 G_p_now = msgToPose3(msg);
    node_odom_[symbol.key()] = G_p_now;

    Pose3 relp = node_odom_[last_odom_node->key()].between(G_p_now);

    // TODO use more accurate relative covariance information...
    // Eigen::Matrix6d cov;
    double sigma_p = 0.02;
    double sigma_q = 0.0025;
    Eigen::VectorXd sigmas(6);
    sigmas << sigma_q, sigma_q, sigma_q, sigma_p, sigma_p, sigma_p;
    Eigen::MatrixXd cov = sigmas.array().pow(2).matrix().asDiagonal();

    // Node that here for the initial estimate, we don't want to use the current odometry estimate.
    // We just want to use the odometry *delta* and update the previous optimized estimate
    Pose3 prev_state = last_odom_node->pose();
    Pose3 current_estimate = prev_state * relp;

    auto node = util::allocate_aligned<SE3Node>(symbol, msg.header.stamp);
    node->pose() = current_estimate;


    auto fac = util::allocate_aligned<CeresBetweenFactor>(last_odom_node,
                                                            node,
                                                            relp,
                                                            cov);

    graph_->addNode(node);
    graph_->addFactor(fac);
    graph_->setModified();

    // ROS_INFO_STREAM("Added new odometry node " << DefaultKeyFormatter(symbol));

    // ROS_INFO_STREAM("Added spine node " << DefaultKeyFormatter(symbol) << " at time = " 
    //         << *node->time() << " (requested t = " << time << ")");


    return node;
}

void OdometryHandler::update() {
    // Update the spine if our messages are exceeding our max period without a node
    nav_msgs::Odometry last_msg;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (msg_queue_.size() < 2) return;
        last_msg = msg_queue_.back();
    }

    // ROS_INFO_STREAM("Msg queue size: " << msg_queue_.size());

    SE3NodePtr last_odom_node = graph_->findLastNode<SE3Node>(node_chr_);


    if (!last_odom_node) {
        // No "spine" yet -- add if the time spanned by the messages currently in the queue 
        // is greater than the max period
        ros::Duration queue_time = last_msg.header.stamp - msg_queue_.front().header.stamp;
        if (queue_time > max_node_period_) {
            // attachSpineNode(msg_queue_.front().header.stamp + max_node_period_);
        }
    } else {
        ros::Duration time_since_node = last_msg.header.stamp - *last_odom_node->time();

        if (time_since_node > max_node_period_) {
            // attachSpineNode(*last_odom_node->time() + max_node_period_);
        }
    }

}