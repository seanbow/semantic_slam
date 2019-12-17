
#include "semantic_slam/LoopCloser.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/SemanticMapper.h"

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
    current_graph_->solver_options().max_solver_time_in_seconds = 5;

    // current_graph_->solver_options().minimizer_type = ceres::LINE_SEARCH;
    // current_graph_->solver_options().max_num_iterations = 100000;

    current_graph_->solver_options().linear_solver_type = ceres::CGNR;
    current_graph_->solver_options().max_linear_solver_iterations = 25;

    // current_graph_->solver_options().linear_solver_type =
    //   ceres::SPARSE_NORMAL_CHOLESKY;

    time_start_ = std::chrono::high_resolution_clock::now();

    // TESTING unfreeze it all
    // Key origin_kf_key = mapper_->getKeyframeByIndex(0)->key();
    // for (auto key_node : current_graph_->nodes()) {
    //     if (key_node.first != origin_kf_key) {
    //         current_graph_->setNodeVariable(key_node.second);
    //     }
    // }

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
    auto last_node = current_graph_->findLastNode<SE3Node>(
      mapper_->getKeyframeByIndex(0)->chr());

    SemanticKeyframe::Ptr closing_kf =
      mapper_->getKeyframeByIndex(last_node->index());

    Pose3 map_T_old_kf = closing_kf->pose();
    Pose3 map_T_new_kf = last_node->pose();

    map_T_old_kf.rotation().normalize();
    map_T_new_kf.rotation().normalize();

    Pose3 old_est_T_new_est = map_T_old_kf.inverse() * map_T_new_kf;
    old_est_T_new_est.rotation().normalize();

    // int earliest_index = std::numeric_limits<int>::max();
    // for (auto& obj : closing_kf->visible_objects()) {
    //     for (auto& kf : obj->keyframe_observations()) {
    //         earliest_index = std::min(earliest_index, kf->index());
    //     }
    // }

    // // Accumulate the list of objects we need to update, i.e. objects that
    // // were visible in the loop
    // std::unordered_set<int> objects_to_update;
    // for (int kf_index = earliest_index; kf_index <= loop_index_; ++kf_index)
    // {
    //     auto kf = mapper_->getKeyframeByIndex(kf_index);

    //     for (auto& obj : kf->visible_objects()) {
    //         objects_to_update.insert(obj->id());
    //     }
    // }

    // Perform the update!
    // Begin with keyframe poses
    for (size_t i = 1; i < mapper_->keyframes().size(); ++i) {
        auto kf = mapper_->getKeyframeByIndex(i);
        if (current_graph_->containsNode(kf->key())) {
            kf->pose() = current_graph_->getNode<SE3Node>(kf->key())->pose();
            kf->pose().rotation().normalize();
        } else {
            auto kf = mapper_->getKeyframeByIndex(i);
            kf->pose() = kf->pose() * old_est_T_new_est;
        }
    }

    // now objects
    for (auto& obj : mapper_->estimated_objects()) {
        if (current_graph_->containsNode(obj->key())) {
            obj->commitGraphSolution(current_graph_);
        } else {
            obj->applyTransformation(old_est_T_new_est);
        }
    }

    // Finally propagate the loop closure delta through to keyframes and objects
    // that have been added past the point of loop closure
    // TODO

    // for (size_t i = loop_index_ + 1; i < mapper_->keyframes().size(); ++i) {
    //     auto kf = mapper_->getKeyframeByIndex(i);

    //     kf->pose() = kf->pose() * old_est_T_new_est;
    // }

    // for (auto& obj : mapper_->estimated_objects()) {
    //     // Note that if an object was seen before the loop, it will still be
    //     // contained in the current graph even if it was held constant in the
    //     // optimization. so this is a valid way to check for objects that
    //     were
    //     // added after loop closure.
    //     if (!current_graph_->containsNode(obj->key())) {
    //         obj->applyTransformation(old_est_T_new_est);
    //     }
    // }

    // Check for map merge??

    return true;
}