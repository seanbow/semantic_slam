#include "semantic_slam/SemanticSmoother.h"

#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/GeometricFeatureHandler.h"
#include "semantic_slam/LoopCloser.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/keypoints/EstimatedKeypoint.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <thread>
#include <unordered_set>

#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/Values.h>

SemanticSmoother::SemanticSmoother(ObjectParams params, SemanticMapper* mapper)
  : mapper_(mapper)
  , params_(params)
  , invalidate_optimization_(false)
  , running_(false)
{
    graph_ = util::allocate_aligned<FactorGraph>();
    essential_graph_ = util::allocate_aligned<FactorGraph>();

    graph_->solver_options().function_tolerance = 1e-4;
    graph_->solver_options().gradient_tolerance = 1e-8;
    graph_->solver_options().parameter_tolerance = 1e-6;

    graph_->solver_options().max_solver_time_in_seconds =
      max_optimization_time_;

    graph_->solver_options().linear_solver_type = ceres::CGNR;
    graph_->solver_options().nonlinear_conjugate_gradient_type =
      ceres::POLAK_RIBIERE;
    graph_->solver_options().max_linear_solver_iterations = 50;

    graph_->solver_options().num_threads = 4;

    if (params_.use_manual_elimination_ordering) {
        graph_->solver_options().linear_solver_ordering =
          std::make_shared<ceres::ParameterBlockOrdering>();
    }

    essential_graph_->setSolverOptions(graph_->solver_options());

    last_kf_covariance_ = Eigen::MatrixXd::Zero(6, 6);
    last_optimized_kf_index_ = 0;

    // will become true if the user sets the handler later
    include_geometric_features_ = false;
}

void
SemanticSmoother::setOrigin(SemanticKeyframe::Ptr origin_frame)
{
    origin_frame->addToGraph(graph_);
    graph_->setNodeConstant(origin_frame->graph_node());

    origin_frame->addToGraph(essential_graph_);
    essential_graph_->setNodeConstant(origin_frame->graph_node());
}

void
SemanticSmoother::start()
{
    running_ = true;

    work_thread_ =
      std::thread(&SemanticSmoother::processingThreadFunction, this);
}

void
SemanticSmoother::join()
{
    work_thread_.join();
}

void
SemanticSmoother::processingThreadFunction()
{
    while (running_) {
        auto new_frames = addNewOdometryToGraph();

        if (graph_->modified() &&
            mapper_->operation_mode() !=
              SemanticMapper::OperationMode::LOOP_CLOSING) {

            if (include_geometric_features_)
                processGeometricFeatureTracks(new_frames);

            tryAddObjectsToGraph();
            freezeNonCovisible(new_frames);

            // Check if a loop closing frame was added to the graph this time...
            bool loop_closure_added = false;
            int loop_closure_index = -1;
            if (mapper_->operation_mode() ==
                SemanticMapper::OperationMode::LOOP_CLOSURE_PENDING) {
                for (auto& frame : new_frames) {
                    if (frame->loop_closing()) {
                        loop_closure_added = true;
                        loop_closure_index = frame->index();
                        break;
                    }
                }
            }

            // if loop_closure_added is true now, we've detected a loop closure
            // and added this loop to the graph for the first time. start the
            // actual loop closing process
            if (loop_closure_added) {
                prepareGraphNodes();
                loop_closer_->startLoopClosing(graph_, loop_closure_index);
                mapper_->setOperationMode(
                  SemanticMapper::OperationMode::LOOP_CLOSING);
            } else {
                if (tryOptimize()) {
                    if (needToComputeCovariances())
                        computeLatestCovariance();
                }
            }

        } else {
            ros::Duration(0.001).sleep();
        }
    }
}

void
SemanticSmoother::stop()
{
    ROS_WARN("Stopping SemanticSmoother");
    running_ = false;
}

void
SemanticSmoother::processGeometricFeatureTracks(
  const std::vector<SemanticKeyframe::Ptr>& new_keyframes)
{
    for (auto& kf : new_keyframes) {
        geom_handler_->addKeyframe(kf);
    }

    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> geom_map_lock(mapper_->geometric_map_mutex(),
                                               std::defer_lock);
    std::lock(graph_lock, geom_map_lock);

    geom_handler_->processPendingFrames();
}

void
SemanticSmoother::tryAddObjectsToGraph()
{
    if (!params_.include_objects_in_graph)
        return;

    std::lock_guard<std::mutex> lock(graph_mutex_);

    for (auto& obj : mapper_->estimated_objects()) {
        if (obj->inGraph()) {
            obj->updateGraphFactors();
        } else if (obj->readyToAddToGraph()) {
            obj->addToGraph();
        }
    }
}

bool
SemanticSmoother::tryOptimize()
{
    if (!graph_->modified())
        return false;

    TIME_TIC;

    invalidate_optimization_ = false;

    prepareGraphNodes();

    bool solve_succeeded = solveGraph();

    if (solve_succeeded && !invalidate_optimization_) {
        commitGraphSolution();
        ROS_INFO_STREAM(
          fmt::format("Solved {} nodes and {} edges in {:.2f} ms.",
                      graph_->num_nodes(),
                      graph_->num_factors(),
                      TIME_TOC));
        last_optimized_kf_index_ = mapper_->getLastKeyframeInGraph()->index();
        return true;
    } else if (!solve_succeeded) {
        ROS_INFO_STREAM("Graph solve failed");
    }

    // No matter what we want this invalidation to be false here. It is only
    // meant to check for loop closures that happen between calls to
    // solveGraph() and commitGraphSolution().
    invalidate_optimization_ = false;
    return false;
}

bool
SemanticSmoother::needToComputeCovariances()
{
    if (mapper_->getLastKeyframeInGraph()->time() - last_kf_covariance_time_ >
        ros::Duration(covariance_delay_)) {
        return true;
    }

    return false;
}

void
SemanticSmoother::setGeometricFeatureHandler(
  boost::shared_ptr<GeometricFeatureHandler> geom)
{
    include_geometric_features_ = true;
    geom_handler_ = geom;
}

void
SemanticSmoother::setLoopCloser(boost::shared_ptr<LoopCloser> closer)
{
    loop_closer_ = closer;
}

bool
SemanticSmoother::computeLatestCovariance()
{
    SemanticKeyframe::Ptr frame = mapper_->getLastKeyframeInGraph();

    bool succeeded = computeCovariances({ frame });

    if (succeeded) {
        last_kf_covariance_ = frame->covariance();
        last_kf_covariance_time_ = frame->time();

        // propagate this covariance forward to keyframes not yet in the
        // graph...
        std::lock_guard<std::mutex> map_lock(mapper_->map_mutex());
        for (size_t i = frame->index() + 1; i < mapper_->keyframes().size();
             ++i) {
            mapper_->keyframes()[i]->covariance() = frame->covariance();
        }
    }

    return succeeded;
}

bool
SemanticSmoother::computeCovariances(
  const std::vector<SemanticKeyframe::Ptr>& frames)
{
    // Need to ensure the graph is unfrozen to get accurate covariance
    // information, otherwise e.g. the first unfrozen frame will have a
    // near-zero covariance
    unfreezeAll();

    bool success = computeCovariancesWithGtsam(frames);

    for (auto& frame : frames) {
        frame->covariance_computed_exactly() = true;
    }

    return success;
}

bool
SemanticSmoother::computeCovariancesWithGtsam(
  const std::vector<SemanticKeyframe::Ptr>& frames)
{
    TIME_TIC;

    boost::shared_ptr<gtsam::NonlinearFactorGraph> gtsam_graph;
    boost::shared_ptr<gtsam::Values> gtsam_values;

    {
        std::lock_guard<std::mutex> lock(graph_mutex_);

        gtsam_graph = graph_->getGtsamGraph();
        gtsam_values = graph_->getGtsamValues();
    }

    // Anchor the origin
    auto origin_factor =
      util::allocate_aligned<gtsam::NonlinearEquality<gtsam::Pose3>>(
        mapper_->getKeyframeByIndex(0)->key(),
        gtsam_values->at<gtsam::Pose3>(symbol_shorthand::X(0)));

    gtsam_graph->push_back(origin_factor);

    try {
        gtsam::Marginals marginals(*gtsam_graph, *gtsam_values);

        for (auto& frame : frames) {
            auto cov = marginals.marginalCovariance(frame->key());

            std::lock_guard<std::mutex> map_lock(mapper_->map_mutex());
            frame->covariance() = cov;
        }

        ROS_INFO_STREAM("Covariance computation took " << TIME_TOC << " ms.");
    } catch (gtsam::IndeterminantLinearSystemException& e) {
        ROS_WARN_STREAM("Covariance computation failed! Error: " << e.what());

        // If this was detected near a semantic object, remove it from the
        // estimation
        Symbol bad_variable(e.nearbyVariable());

        std::lock_guard<std::mutex> lock(mapper_->map_mutex());

        if (bad_variable.chr() == 'c') {
            auto obj = mapper_->getObjectByIndex(bad_variable.index());
            if (obj)
                obj->removeFromEstimation();
        } else if (bad_variable.chr() == 'l') {
            auto obj = mapper_->getObjectByKeypointKey(bad_variable.key());
            if (obj)
                obj->removeFromEstimation();
        } else if (bad_variable.chr() == 'o') {
            auto obj = mapper_->getObjectByKey(bad_variable.key());
            if (obj)
                obj->removeFromEstimation();
        }

        // and same for geometric landmarks
        if (bad_variable.chr() == 'g') {
            geom_handler_->removeLandmark(bad_variable.index());
        }

        return false;
    }

    return true;
}

void
SemanticSmoother::prepareGraphNodes()
{
    std::unique_lock<std::mutex> map_lock(mapper_->map_mutex(),
                                          std::defer_lock);
    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::lock(map_lock, graph_lock);

    for (auto& kf : mapper_->keyframes()) {
        if (kf->inGraph()) {
            kf->graph_node()->pose() = kf->pose();
        }
    }

    for (auto& obj : mapper_->estimated_objects()) {
        if (obj->inGraph()) {
            obj->prepareGraphNode();
        }
    }
}

void
SemanticSmoother::commitGraphSolution()
{
    std::unique_lock<std::mutex> map_lock(mapper_->map_mutex(),
                                          std::defer_lock);
    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::lock(map_lock, graph_lock);

    // Find the latest keyframe in the graph optimization to check the computed
    // transformation
    SemanticKeyframe::Ptr last_in_graph = mapper_->getLastKeyframeInGraph();

    const Pose3& old_pose = last_in_graph->pose();

    // if (params_.optimization_backend == OptimizationBackend::CERES) {
    Pose3 new_pose = last_in_graph->graph_node()->pose();
    new_pose.rotation().normalize();

    Pose3 old_T_new = old_pose.inverse() * new_pose;
    old_T_new.rotation().normalize();

    for (auto& kf : mapper_->keyframes()) {
        if (kf->inGraph()) {
            kf->pose() = kf->graph_node()->pose();
            kf->pose().rotation().normalize();
        } else {
            // Keyframes not yet in the graph will be later so just
            // propagate the computed transform forward
            kf->pose() = kf->pose() * old_T_new;
        }
    }

    for (auto& obj : mapper_->estimated_objects()) {
        if (obj->inGraph()) {
            obj->commitGraphSolution();
        } else if (!obj->bad()) {
            // Update the object based on recomputed camera poses
            obj->optimizeStructure();
        }
    }
}

std::vector<SemanticKeyframe::Ptr>
SemanticSmoother::addNewOdometryToGraph()
{
    std::vector<SemanticKeyframe::Ptr> new_frames;

    std::lock_guard<std::mutex> lock(graph_mutex_);

    for (auto& kf : mapper_->keyframes()) {
        if (!kf->inGraph() && kf->measurements_processed()) {
            kf->addToGraph(graph_);
            kf->addToGraph(essential_graph_);
            new_frames.push_back(kf);
        }
    }

    return new_frames;
}

void
SemanticSmoother::freezeNonCovisible(
  const std::vector<SemanticKeyframe::Ptr>& target_frames)
{
    // Iterate over the target frames and their covisible frames, collecting the
    // frames and objects that will remain unfrozen in the graph

    // In case of a loop closure there may be frames i and j are covisible but
    // some frame i < k < j is not covisible with i or j. So we need to make
    // sure that we unfreeze these intermediate frames as well.

    int min_frame = std::numeric_limits<int>::max();
    int max_frame = std::numeric_limits<int>::lowest();
    int min_frame_in_targets = min_frame;

    for (const auto& frame : target_frames) {
        min_frame = std::min(min_frame, frame->index());
        max_frame = std::max(max_frame, frame->index());
        min_frame_in_targets = std::min(min_frame_in_targets, frame->index());

        for (const auto& cov_frame : frame->neighbors()) {
            min_frame = std::min(min_frame, cov_frame.first->index());
            max_frame = std::max(max_frame, cov_frame.first->index());
        }
    }

    // Give ourself a smoothing lag...
    // TODO compute this based on geometric features?
    min_frame = std::min(min_frame, min_frame_in_targets - smoothing_length_);
    min_frame = std::max(min_frame, 0);

    unfrozen_kfs_.clear();
    unfrozen_objs_.clear();

    for (int i = min_frame; i <= max_frame; ++i) {
        unfrozen_kfs_.insert(i);

        for (auto& obj : mapper_->keyframes()[i]->visible_objects()) {
            unfrozen_objs_.insert(obj->id());
        }
    }

    for (const auto& kf : mapper_->keyframes()) {
        if (!kf->inGraph())
            continue;

        if (unfrozen_kfs_.count(kf->index())) {
            graph_->setNodeVariable(kf->graph_node());
            essential_graph_->setNodeVariable(kf->graph_node());
        } else {
            graph_->setNodeConstant(kf->graph_node());
            essential_graph_->setNodeConstant(kf->graph_node());
        }
    }

    // now objects
    for (const auto& obj : mapper_->estimated_objects()) {
        if (!obj->inGraph())
            continue;

        if (unfrozen_objs_.count(obj->id())) {
            obj->setVariableInGraph();
        } else {
            obj->setConstantInGraph();
        }
    }
}

void
SemanticSmoother::unfreezeAll()
{
    std::lock_guard<std::mutex> lock(graph_mutex_);

    for (const auto& kf : mapper_->keyframes()) {
        if (!kf->inGraph())
            continue;

        // do NOT unfreeze the first (gauge freedom)
        if (kf->index() > 0) {
            graph_->setNodeVariable(kf->graph_node());
            essential_graph_->setNodeVariable(kf->graph_node());
        }
    }

    for (const auto& obj : mapper_->estimated_objects()) {
        if (!obj->inGraph())
            continue;

        obj->setVariableInGraph();
    }
}

bool
SemanticSmoother::solveGraph()
{
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return graph_->solve(false);
}

void
SemanticSmoother::setVerbose(bool verbose_optimization)
{
    verbose_optimization_ = verbose_optimization;
}

void
SemanticSmoother::setCovarianceDelay(double covariance_delay)
{
    covariance_delay_ = covariance_delay;
}

void
SemanticSmoother::setMaxOptimizationTime(double max_optimization_time)
{
    max_optimization_time_ = max_optimization_time;

    graph_->solver_options().max_solver_time_in_seconds =
      max_optimization_time_;
}

void
SemanticSmoother::setSmoothingLength(int smoothing_length)
{
    smoothing_length_ = smoothing_length;
}

void
SemanticSmoother::setLoopClosureThreshold(int loop_closure_threshold)
{
    loop_closure_threshold_ = loop_closure_threshold;
}