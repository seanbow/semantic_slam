
#include "semantic_slam/LoopCloser.h"
#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <unordered_set>

LoopCloser::LoopCloser(SemanticMapper* mapper)
  : mapper_(mapper)
  , solve_succeeded_(false)
{}

void
LoopCloser::startLoopClosing(boost::shared_ptr<FactorGraph> graph,
                             int loop_closing_kf_index)
{
    current_graph_ = graph->clone();
    loop_index_ = loop_closing_kf_index;

    running_ = true;
    thread_ = std::thread(&LoopCloser::optimizeCurrentGraph, this);
    thread_.detach();
}

void
LoopCloser::optimizeCurrentGraph()
{
    current_graph_->solver_options().max_solver_time_in_seconds = 4;

    // current_graph_->solver_options().minimizer_type = ceres::LINE_SEARCH;
    // current_graph_->solver_options().line_search_direction_type =
    //   ceres::NONLINEAR_CONJUGATE_GRADIENT;
    // current_graph_->solver_options().nonlinear_conjugate_gradient_type =
    //   ceres::POLAK_RIBIERE;

    current_graph_->solver_options().max_num_iterations = 100000;

    current_graph_->solver_options().linear_solver_type = ceres::CGNR;
    current_graph_->solver_options().nonlinear_conjugate_gradient_type =
      ceres::POLAK_RIBIERE;
    current_graph_->solver_options().max_linear_solver_iterations = 25;

    // current_graph_->solver_options().linear_solver_type =
    //   ceres::SPARSE_NORMAL_CHOLESKY;

    current_graph_->solver_options().function_tolerance = 1e-4;
    current_graph_->solver_options().gradient_tolerance = 1e-8;
    current_graph_->solver_options().parameter_tolerance = 1e-6;

    time_start_ = std::chrono::high_resolution_clock::now();

    solve_succeeded_ = current_graph_->solve(true);

    running_ = false;
    time_end_ = std::chrono::high_resolution_clock::now();
}

bool
LoopCloser::running()
{
    return running_;
}

bool
LoopCloser::containsNode(Key key)
{
    return current_graph_->containsNode(key);
}

bool
LoopCloser::updateLoopInMapper()
{
    // Take our computed solution and update all objects and keyframes
    // in the SemanticMapper
    if (running())
        return false;

    std::chrono::duration<double, std::micro> duration =
      time_end_ - time_start_;
    ROS_INFO_STREAM(fmt::format("Loop closure optimization completed in {} ms.",
                                duration.count() / 1000.0));

    // Rather than try to determing the beginning of the loop, just update all
    // poses. The poses before the loop will be frozen and it's cheap to do the
    // update

    // auto last_node = current_graph_->findLastNode<SE3Node>(
    //   mapper_->getKeyframeByIndex(0)->chr());
    // SemanticKeyframe::Ptr closing_kf =
    //   mapper_->getKeyframeByIndex(last_node->index());
    // Pose3 map_T_old_kf = closing_kf->pose();
    // Pose3 map_T_new_kf = last_node->pose();

    auto closing_kf = mapper_->getKeyframeByIndex(loop_index_);
    Pose3 new_pose =
      current_graph_->getNode<SE3Node>(closing_kf->key())->pose();
    Pose3 old_pose = closing_kf->pose();

    new_pose.rotation().normalize();

    Pose3 old_T_new = old_pose.inverse() * new_pose;
    old_T_new.rotation().normalize();

    // Perform the update!
    // Begin with keyframe poses
    for (size_t i = 1; i < mapper_->keyframes().size(); ++i) {
        auto kf = mapper_->getKeyframeByIndex(i);
        if (current_graph_->containsNode(kf->key())) {
            kf->pose() = current_graph_->getNode<SE3Node>(kf->key())->pose();
            kf->pose().rotation().normalize();
        } else {
            auto kf = mapper_->getKeyframeByIndex(i);
            kf->pose() = kf->pose() * old_T_new;
        }
    }

    // now objects
    for (auto& obj : mapper_->estimated_objects()) {
        if (current_graph_->containsNode(obj->key())) {
            obj->commitGraphSolution(current_graph_);
        } else if (!obj->bad()) {
            // obj->applyTransformation(old_est_T_new_est);
            obj->optimizeStructure();
        }
    }

    // Check for map merge??

    return true;
}