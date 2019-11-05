#include "semantic_slam/FactorGraph.h"

// #include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

FactorGraph::FactorGraph()
  : modified_(false)
{
    graph_ = util::allocate_aligned<CeresFactorGraph>();
    // values_ = boost::make_shared<gtsam::Values>();
}


void FactorGraph::addFactor(CeresFactorPtr fac)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);

        factors_.push_back(fac);
        graph_->addFactor(fac);
    }

    // do *not* set modified here -- let the calling handler set it themselves
    // so they can finish their full set of modifications first
}

CeresNodePtr
FactorGraph::getNode(Symbol sym)
{
    for (auto& node : nodes_) {
        if (node->symbol() == sym) {
            return node;
        }
    }

    return nullptr;
}

bool FactorGraph::solve(bool verbose)
{
    graph_->solve(verbose);
    /*
    gtsam::LevenbergMarquardtParams lm_params;
    // lm_params.setVerbosityLM("SUMMARY");
    // lm_params.setVerbosityLM("DAMPED");
    lm_params.diagonalDamping = true;
    lm_params.relativeErrorTol = 1.0e-2;
    lm_params.absoluteErrorTol = 1.0e-2;

    try {
        std::lock_guard<std::mutex> lock(mutex_);
        gtsam::LevenbergMarquardtOptimizer optimizer(*graph_, *values_, lm_params);
        *values_ = optimizer.optimize();
    } catch (std::exception& e) {
        ROS_WARN("Error when optimizing factor graph");
        ROS_WARN_STREAM(e.what());
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(mutex_);
        marginals_ = boost::make_shared<gtsam::Marginals>(*graph_, *values_);
    } catch (std::exception& e) {
        ROS_WARN("Error when computing marginals");
        ROS_WARN_STREAM(e.what());
        // should we return false here?? probably not
    }
    */

    modified_ = false;

    return true;

}

// bool FactorGraph::marginalCovariance(gtsam::Key key, Eigen::MatrixXd& cov)
// {
//     try {
//         std::lock_guard<std::mutex> lock(mutex_);
//         cov = marginals_->marginalCovariance(key);
//         return true;
//     } catch (std::exception& e) {
//         ROS_WARN_STREAM("Failed to get covariance.");
//         ROS_WARN_STREAM(e.what());
//         return false;
//     }
// }

CeresNodePtr
FactorGraph::findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0) return nullptr;

    ros::Time last_time(0);
    bool found = false;
    size_t node_index = 0;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i]->chr() != symbol_chr) continue;

        if (!nodes_[i]->time()) continue;

        if (nodes_[i]->time() > last_time && nodes_[i]->time() <= time) {
            last_time = *nodes_[i]->time();
            found = true;
            node_index = i;
        }
    }

    if (found) return nodes_[node_index];
    else return nullptr;
}


CeresNodePtr
FactorGraph::findFirstNodeAfterTime(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0) return nullptr;

    ros::Time first_time = ros::TIME_MAX;
    bool found = false;
    size_t node_index = 0;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i]->chr() != symbol_chr) continue;

        if (!nodes_[i]->time()) continue;

        if (nodes_[i]->time() <= first_time && nodes_[i]->time() >= time) {
            first_time = *nodes_[i]->time();
            found = true;
            node_index = i;
        }
    }

    if (found) return nodes_[node_index];
    else return nullptr;
}

CeresNodePtr
FactorGraph::findLastNode(unsigned char symbol_chr)
{
    CeresNodePtr result = nullptr;

    ros::Time last_time = ros::Time(0);

    for (auto& node : nodes_) {
        if (node->chr() != symbol_chr) continue;

        if (!node->time()) continue;

        if (node->time() > last_time) {
            last_time = *node->time();
            result = node;
        }
    }

    return result;
}