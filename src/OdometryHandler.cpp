#include "semantic_slam/OdometryHandler.h"

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>


using namespace std::string_literals;

void OdometryHandler::setup()
{
    ROS_INFO("Starting odometry handler.");
    std::string odometry_topic;
    pnh_.param("odom_topic", odometry_topic, "/odom"s);

    ROS_INFO_STREAM("Subscribing to topic " << odometry_topic);

    subscriber_ = nh_.subscribe(odometry_topic, 1000, &OdometryHandler::msgCallback, this);

    received_msgs_ = 0;

    node_chr_ = 'x';

    node_period_ = ros::Duration(0); // seconds
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

    cv_->notify_all();
}

gtsam::Pose3 OdometryHandler::msgToPose3(const nav_msgs::Odometry& msg)
{
    auto p_msg = msg.pose.pose.position;
    auto q_msg = msg.pose.pose.orientation;

    gtsam::Point3 p(p_msg.x, p_msg.y, p_msg.z);
    Eigen::Quaterniond q;
    q.x() = q_msg.x;
    q.y() = q_msg.y;
    q.z() = q_msg.z;
    q.w() = q_msg.w;

    gtsam::Pose3 G_p(gtsam::Rot3(q), p);

    return G_p;
}

void OdometryHandler::update() {

    if (msg_queue_.empty()) return;    
    
    NodeInfoPtr last_odom_node = graph_->findLastNode(node_chr_);

    bool is_first_node;
    gtsam::Symbol symbol;

    if (last_odom_node) {
        is_first_node = false;
        symbol = gtsam::Symbol(node_chr_, last_odom_node->index() + 1);
    } else {
        is_first_node = true;
        symbol = gtsam::Symbol(node_chr_, 0);
    }

    auto msg = msg_queue_.front();

    {
        std::lock_guard<std::mutex> guard(mutex_);
        msg_queue_.pop_front();

        if (!is_first_node) {
            // wait until we have waited enough time since the last odometry node we added
            while (!msg_queue_.empty() && msg.header.stamp - last_time_ < node_period_) {
                msg = msg_queue_.front();
                msg_queue_.pop_front();
            }

            // check that we got the right message not that we just expended the queue
            if (msg.header.stamp - last_time_ < node_period_) {
                return;
            }
        }
    }

    last_time_ = msg.header.stamp;


    gtsam::Pose3 G_p_now = msgToPose3(msg);

    if (is_first_node) {
        // anchor origin and return
        gtsam::Pose3 origin;

        // TODO figure out best values for origin anchoring factor noise
        double sigma_p = .01;
        double sigma_q = .001;
        Eigen::VectorXd sigmas(6);
        sigmas << sigma_q, sigma_q, sigma_q, sigma_p, sigma_p, sigma_p;

        gtsam::noiseModel::Gaussian::shared_ptr anchor_cov = 
            gtsam::noiseModel::Diagonal::Sigmas(sigmas);

        auto gtsam_fac = util::allocate_aligned<gtsam::PriorFactor<gtsam::Pose3>>(symbol, origin, anchor_cov);
        auto fac_info = FactorInfo::Create(FactorType::ORIGIN, gtsam_fac, 0);

        auto node = NodeInfo::Create(symbol, msg.header.stamp);

        graph_->addNode(node, origin);
        graph_->addFactor(fac_info);

        ROS_INFO_STREAM("Added first node " << gtsam::DefaultKeyFormatter(symbol) 
                            << " to graph, origin = " << origin);

    } else {
        // create a between constraint between the subsequent odometry nodes
        gtsam::Pose3 relp = last_odom_.between(G_p_now);

        // TODO use more accurate relative covariance information...
        // Eigen::Matrix6d cov;
        double sigma_p = 0.01;
        double sigma_q = 0.001;
        Eigen::VectorXd sigmas(6);
        sigmas << sigma_q, sigma_q, sigma_q, sigma_p, sigma_p, sigma_p;
        auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

        auto gtsam_fac = util::allocate_aligned<gtsam::BetweenFactor<gtsam::Pose3>>(last_odom_node->symbol(),
                                                                                    symbol,
                                                                                    relp,
                                                                                    odometry_noise);
        auto fac_info = FactorInfo::Create(FactorType::ODOMETRY, gtsam_fac, symbol.index());
        auto node = NodeInfo::Create(symbol, msg.header.stamp);

        // Node that here for the initial estimate, we don't want to use the current odometry estimate.
        // We just want to use the odometry *delta* and update the previous optimized estimate
        gtsam::Pose3 prev_state, current_estimate;
        bool succeeded = graph_->getEstimate(last_odom_node->symbol(), prev_state);

        if (!succeeded) {
            ROS_ERROR_STREAM("Failed to get state estimate for node " 
                                << gtsam::DefaultKeyFormatter(last_odom_node->symbol()));
            current_estimate = G_p_now; // fallback to odometry
        } else {
            current_estimate = relp * prev_state;
        }

        graph_->addNode(node, current_estimate);
        graph_->addFactor(fac_info);

        ROS_INFO_STREAM("Added new odometry node " << gtsam::DefaultKeyFormatter(symbol));
        // ROS_INFO_STREAM("Added between factor, between " << gtsam::DefaultKeyFormatter(last_odom_node->symbol())
        //                     << " and " << gtsam::DefaultKeyFormatter(symbol) << ", = \n" << relp);
    }

    graph_->setModified();

    last_odom_ = G_p_now;

    // gtsam::Pose3 between = last_odom_.between(G_p_now);

    // ROS_INFO_STREAM("Between = " << between);
	// // gtsam::noiseModel::Gaussian::shared_ptr cov = gtsam::noiseModel::Gaussian::Covariance(relpose_covariance_);

    // gtsam::NonlinearFactorGraph G;
    // gtsam::Values values;

    // gtsam::Pose3 last_pose_est = graph_->calculateEstimate<gtsam::Pose3>(sym::X(pose_id - 1));

    // gtsam::Pose3 relp = last_pose_.between(G_p_now);

    // values.insert(sym::X(pose_id), relp * last_pose_est);

    // auto fac = util::allocate_aligned<gtsam::BetweenFactor<gtsam::Pose3>>(sym::X(pose_id - 1), 
    //                                                                         sym::X(pose_id), 
    //                                                                         relp, 
    //                                                                         cov);

    // auto fac_info = boost::make_shared<FactorInfo>(FactorType::ODOMETRY, pose_id, fac);

    // graph_->isamUpdate({fac_info}, values);

    // last_time_ = next_time;
    // last_pose_ = G_p_now;

    // return isam.calculateEstimate();
}