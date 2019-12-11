
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
    current_graph_->solver_options().linear_solver_type = ceres::CGNR;

    bool solved = current_graph_->solve();

    running_ = false;
}

bool
LoopCloser::running()
{
    return running_;
}

bool
LoopCloser::updateLoopInGraph(boost::shared_ptr<FactorGraph> graph)
{
    // Take our computed solution and update all the nodes in the
    // graph passed in here
    if (running_)
        return false;

    // Based on the loop closing index passed in, compute the beginning
    // of the loop as the first keyframe that observed an object
    // also observed in that index.
    SemanticKeyframe::Ptr closing_kf = mapper_->getKeyframeByIndex(loop_index_);

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

    // Check for map merge??
}