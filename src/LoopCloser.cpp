
#include "semantic_slam/LoopCloser.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/SemanticMapper.h"

#include <unordered_set>

LoopCloser::LoopCloser(SemanticMapper* mapper)
  : mapper_(mapper)
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
    current_graph_->solver_options().max_solver_time_in_seconds = 20;
    // current_graph_->solver_options().linear_solver_type = ceres::CGNR;
    current_graph_->solver_options().linear_solver_type =
      ceres::SPARSE_NORMAL_CHOLESKY;

    time_start_ = std::chrono::high_resolution_clock::now();

    // TESTING unfreeze it all
    // Key origin_kf_key = mapper_->getKeyframeByIndex(0)->key();
    // for (auto key_node : current_graph_->nodes()) {
    //     if (key_node.first != origin_kf_key) {
    //         current_graph_->setNodeVariable(key_node.second);
    //     }
    // }

    bool solved = current_graph_->solve(true);

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

    // Based on the loop closing index passed in, compute the beginning
    // of the loop as the first keyframe that observed an object
    // also observed in that index.
    // TODO should we just check which are unfrozen in the graph??
    SemanticKeyframe::Ptr closing_kf = mapper_->getKeyframeByIndex(loop_index_);
    Pose3 old_map_T_kf = closing_kf->pose();

    int earliest_index = std::numeric_limits<int>::max();
    for (auto& obj : closing_kf->visible_objects()) {
        for (auto& kf : obj->keyframe_observations()) {
            earliest_index = std::min(earliest_index, kf->index());
        }
    }

    // Accumulate the list of objects we need to update, i.e. objects that
    // were visible in the loop
    std::unordered_set<int> objects_to_update;
    for (int kf_index = earliest_index; kf_index <= loop_index_; ++kf_index) {
        auto kf = mapper_->getKeyframeByIndex(kf_index);

        for (auto& obj : kf->visible_objects()) {
            objects_to_update.insert(obj->id());
        }
    }

    // Perform the update!
    // Begin with keyframe poses
    for (int i = earliest_index; i <= loop_index_; ++i) {
        auto kf = mapper_->getKeyframeByIndex(i);

        SE3NodePtr updated_node = current_graph_->getNode<SE3Node>(kf->key());

        kf->pose() = updated_node->pose();
    }

    // now objects
    for (auto id : objects_to_update) {
        EstimatedObject::Ptr obj = mapper_->getObjectByIndex(id);
        obj->commitGraphSolution(current_graph_);
    }

    // Finally propagate the loop closure delta through to keyframes and objects
    // that have been added past the point of loop closure
    // TODO
    Pose3 old_est_T_new_est = old_map_T_kf.inverse() * closing_kf->pose();

    for (int i = loop_index_ + 1; i < mapper_->keyframes().size(); ++i) {
        auto kf = mapper_->getKeyframeByIndex(i);

        kf->pose() = kf->pose() * old_est_T_new_est;
    }

    for (auto& obj : mapper_->estimated_objects()) {
        // Note that if an object was seen before the loop, it will still be
        // contained in the current graph even if it was held constant in the
        // optimization. so this is a valid way to check for objects that were
        // added after loop closure.
        if (!current_graph_->containsNode(obj->key())) {
            obj->applyTransformation(old_est_T_new_est);
        }
    }

    // Check for map merge??
}