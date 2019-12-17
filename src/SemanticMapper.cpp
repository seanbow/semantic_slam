
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/ExternalOdometryHandler.h"
#include "semantic_slam/GeometricFeatureHandler.h"
#include "semantic_slam/LoopCloser.h"
#include "semantic_slam/MLDataAssociator.h"
#include "semantic_slam/MultiProjectionFactor.h"
#include "semantic_slam/SE3Node.h"

#include <ros/package.h>

#include <boost/filesystem.hpp>
#include <thread>
#include <unordered_set>

#include <visualization_msgs/MarkerArray.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

using namespace std::string_literals;
namespace sym = symbol_shorthand;

SemanticMapper::SemanticMapper()
  : nh_()
  , pnh_("~")
{
    setup();
}

void
SemanticMapper::setup()
{
    ROS_INFO("Starting object handler.");
    std::string object_topic;
    pnh_.param(
      "keypoint_detection_topic", object_topic, "/semslam/img_keypoints"s);

    ROS_INFO_STREAM("[SemanticMapper] Subscribing to topic " << object_topic);

    subscriber_ =
      nh_.subscribe(object_topic, 1000, &SemanticMapper::msgCallback, this);

    graph_ = util::allocate_aligned<FactorGraph>();
    essential_graph_ = util::allocate_aligned<FactorGraph>();

    loop_closer_ = util::allocate_aligned<LoopCloser>(this);

    received_msgs_ = 0;
    measurements_processed_ = 0;
    n_landmarks_ = 0;

    last_kf_covariance_ = Eigen::MatrixXd::Zero(6, 6);
    last_optimized_kf_index_ = 0;

    running_ = false;

    node_chr_ = 'o';

    std::string base_path = ros::package::getPath("semantic_slam");
    std::string model_dir = base_path + std::string("/models/objects/");

    loadModelFiles(model_dir);
    loadCalibration();
    loadParameters();

    // solver_options_.trust_region_strategy_type = ceres::DOGLEG;
    // solver_options_.dogleg_type = ceres::SUBSPACE_DOGLEG;

    // solver_options_.function_tolerance = 1e-4;
    // solver_options_.gradient_tolerance = 1e-8;

    // solver_options_.max_num_iterations = 10;

    graph_->solver_options().max_solver_time_in_seconds =
      max_optimization_time_;

    // graph_->solver_options().linear_solver_type = ceres::SPARSE_SCHUR;

    // graph_->solver_options().linear_solver_type = ceres::ITERATIVE_SCHUR;
    // graph_->solver_options().preconditioner_type = ceres::SCHUR_JACOBI;
    // graph_->solver_options().use_explicit_schur_complement = true;

    graph_->solver_options().linear_solver_type = ceres::CGNR;
    graph_->solver_options().num_threads = 4;

    if (params_.use_manual_elimination_ordering) {
        graph_->solver_options().linear_solver_ordering =
          std::make_shared<ceres::ParameterBlockOrdering>();
    }

    // graph_->setSolverOptions(solver_options_);
    essential_graph_->setSolverOptions(graph_->solver_options());

    operation_mode_ = OperationMode::NORMAL;
    invalidate_local_optimization_ = false;
}

void
SemanticMapper::setOdometryHandler(
  boost::shared_ptr<ExternalOdometryHandler> odom)
{
    odometry_handler_ = odom;
    odom->setGraph(graph_);
    odom->setMapper(this);
    odom->setEssentialGraph(essential_graph_);
    odom->setup();
}

void
SemanticMapper::setGeometricFeatureHandler(
  boost::shared_ptr<GeometricFeatureHandler> geom)
{
    geom_handler_ = geom;
    geom->setGraph(graph_);
    geom->setMapper(this);
    geom->setEssentialGraph(essential_graph_);
    geom->setExtrinsicCalibration(I_T_C_);
    geom->setup();
}

void
SemanticMapper::start()
{
    running_ = true;

    anchorOrigin();

    // while (ros::ok() && running_) {
    //     updateObjects();
    //     // visualizeObjectMeshes();
    //     for (auto& p : presenters_) p->present(keyframes_,
    //     estimated_objects_); addNewOdometryToGraph(); tryAddObjectsToGraph();

    //     bool did_optimize = tryOptimize();

    //     if (did_optimize) {
    //         computeLandmarkCovariances();
    //     }

    //     ros::Duration(0.003).sleep();
    // }

    std::thread process_messages_thread(
      &SemanticMapper::processMessagesUpdateObjectsThread, this);
    std::thread graph_optimize_thread(
      &SemanticMapper::addObjectsAndOptimizeGraphThread, this);

    process_messages_thread.join();
    graph_optimize_thread.join();

    running_ = false;
}

void
SemanticMapper::processMessagesUpdateObjectsThread()
{
    while (ros::ok() && running_) {

        // bool processed_msg = false;

        if (operation_mode_ == OperationMode::LOOP_CLOSING) {
            checkLoopClosingDone();
        }

        if (haveNextKeyframe()) {

            SemanticKeyframe::Ptr next_keyframe = tryFetchNextKeyframe();

            if (next_keyframe) {

                pending_keyframes_.push_back(next_keyframe);

                processPendingKeyframes();

                std::unique_lock<std::mutex> lock(map_mutex_, std::defer_lock);
                std::unique_lock<std::mutex> present_lock(present_mutex_,
                                                          std::defer_lock);
                std::lock(lock, present_lock);

                for (auto& p : presenters_)
                    p->present(keyframes_, estimated_objects_);
            }

        } else {
            ros::Duration(0.001).sleep();
        }
    }

    running_ = false;
}

void
SemanticMapper::addObjectsAndOptimizeGraphThread()
{
    while (ros::ok() && running_) {
        auto new_frames = addNewOdometryToGraph();

        if (graph_->modified() &&
            operation_mode_ != OperationMode::LOOP_CLOSING) {

            if (include_geometric_features_)
                processGeometricFeatureTracks(new_frames);

            tryAddObjectsToGraph();
            freezeNonCovisible(new_frames);

            // Check if a loop closing frame was added to the graph this time...
            bool loop_closure_added = false;
            if (operation_mode_ == OperationMode::LOOP_CLOSURE_PENDING) {
                for (auto& frame : new_frames) {
                    if (frame->loop_closing()) {
                        loop_closure_added = true;
                        break;
                    }
                }
            }

            // if loop_closure_added is true now, we've detected a loop closure
            // and added this loop to the graph for the first time. start the
            // actual loop closing process
            if (loop_closure_added) {
                prepareGraphNodes();
                loop_closer_->startLoopClosing(essential_graph_,
                                               loop_closure_index_);
                operation_mode_ = OperationMode::LOOP_CLOSING;
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

    running_ = false;
}

void
SemanticMapper::processPendingKeyframes()
{
    if (pending_keyframes_.empty())
        return;

    // ros::Time last_added_kf_time_ = pending_keyframes_.front()->time();

    // Use a heuristic to prevent the tracking part from getting too far
    // ahead...
    // TODO think about this more
    // Say we can go until we get half of our smoothing window ahead of the last
    // optimized keyframe
    int last_added_kf_index = pending_keyframes_.front()->index();

    while (!pending_keyframes_.empty() &&
           (last_added_kf_index - last_optimized_kf_index_) <
             smoothing_length_ / 2) {
        auto next_keyframe = pending_keyframes_.front();
        pending_keyframes_.pop_front();

        updateKeyframeObjects(next_keyframe);

        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            next_keyframe->updateConnections();
            next_keyframe->measurements_processed() = true;
        }

        last_added_kf_index = next_keyframe->index();

        // if no objects were detected then earliest_object == INT_MAX and the
        // following will never be true
        if (operation_mode_ == OperationMode::NORMAL &&
            next_keyframe->loop_closing()) {
            ROS_WARN("LOOP CLOSURE!!");

            operation_mode_ = OperationMode::LOOP_CLOSURE_PENDING;
            loop_closure_index_ = next_keyframe->index();

            // because of the asynchronous nature of how measurements are
            // associated and keyframe created (this thread) and how these
            // factors are actually added to the graph (other thread), we can't
            // start the loop closer quite yet as the graph won't contain the
            // loop closing measurements!! need to wait for other thread to
            // incorporate this frame into the graph.

            // loop_closer_->startLoopClosing(essential_graph_,
            //                                next_keyframe->index());
        }
    }
}

bool
SemanticMapper::checkLoopClosingDone()
{
    if (loop_closer_->running())
        return false;

    // Loop closer is done, update our map with its result
    // If we're currently in the middle of a local optimization, it will likely
    // finish after we perform this update, and try to update the graph with
    // stale values. Set an invalidation flag to prevent that from happening
    invalidate_local_optimization_ = true;

    std::lock_guard<std::mutex> map_lock(map_mutex_);

    loop_closer_->updateLoopInMapper();

    operation_mode_ = OperationMode::NORMAL;

    return true;
}

void
SemanticMapper::processGeometricFeatureTracks(
  const std::vector<SemanticKeyframe::Ptr>& new_keyframes)
{
    // TIME_TIC;

    for (auto& kf : new_keyframes) {
        geom_handler_->addKeyframe(kf);
    }

    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> present_lock(present_mutex_, std::defer_lock);
    std::lock(graph_lock, present_lock);

    geom_handler_->processPendingFrames();

    // check covisibility graph...
    // if (new_keyframes.size() > 0) {
    //     std::cout << "Observed features for kf " <<
    //     new_keyframes.front()->index() << ": "; for (auto feat :
    //     new_keyframes.front()->visible_geometric_features()) {
    //         std::cout << feat->id << " ";
    //     }
    //     std::cout << "\nGeom covisibility for kf " <<
    //     new_keyframes.front()->index() << ": "; for (auto neighbor :
    //     new_keyframes.front()->geometric_neighbors()) {
    //         std::cout << neighbor.first->index() << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // ROS_INFO_STREAM("Tracking features took " << TIME_TOC << " ms." <<
    // std::endl);
}

void
SemanticMapper::freezeNonCovisible(
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

        for (auto& obj : keyframes_[i]->visible_objects()) {
            unfrozen_objs_.insert(obj->id());
        }
    }

    for (const auto& kf : keyframes_) {
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
    for (const auto& obj : estimated_objects_) {
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
SemanticMapper::unfreezeAll()
{
    std::lock_guard<std::mutex> lock(graph_mutex_);

    for (const auto& kf : keyframes_) {
        if (!kf->inGraph())
            continue;

        // do NOT unfreeze the first (gauge freedom)
        if (kf->index() > 0) {
            graph_->setNodeVariable(kf->graph_node());
            essential_graph_->setNodeVariable(kf->graph_node());
        }
    }

    for (const auto& obj : estimated_objects_) {
        if (!obj->inGraph())
            continue;

        obj->setVariableInGraph();
    }
}

std::vector<SemanticKeyframe::Ptr>
SemanticMapper::addNewOdometryToGraph()
{
    std::vector<SemanticKeyframe::Ptr> new_frames;

    std::lock_guard<std::mutex> lock(graph_mutex_);

    for (auto& kf : keyframes_) {
        if (!kf->inGraph() && kf->measurements_processed()) {
            kf->addToGraph(graph_);
            kf->addToGraph(essential_graph_);
            new_frames.push_back(kf);
        }
    }

    return new_frames;
}

void
SemanticMapper::tryAddObjectsToGraph()
{
    if (!params_.include_objects_in_graph)
        return;

    std::lock_guard<std::mutex> lock(graph_mutex_);

    for (auto& obj : estimated_objects_) {
        if (obj->inGraph()) {
            obj->updateGraphFactors();
        } else if (obj->readyToAddToGraph()) {
            obj->addToGraph();
        }
    }
}

bool
SemanticMapper::needToComputeCovariances()
{
    if (getLastKeyframeInGraph()->time() - last_kf_covariance_time_ >
        ros::Duration(covariance_delay_)) {
        return true;
    }

    // if (Plxs_.size() != estimated_objects_.size()) {
    //     return true;
    // }

    return false;

    // return true;
}

bool
SemanticMapper::computeLatestCovariance()
{
    SemanticKeyframe::Ptr frame = getLastKeyframeInGraph();

    bool succeeded = computeCovariances({ frame });

    if (succeeded) {
        last_kf_covariance_ = frame->covariance();
        last_kf_covariance_time_ = frame->time();

        // propagate this covariance forward to keyframes not yet in the
        // graph...
        std::lock_guard<std::mutex> map_lock(map_mutex_);
        for (size_t i = frame->index() + 1; i < keyframes_.size(); ++i) {
            keyframes_[i]->covariance() = frame->covariance();
        }
    }

    return succeeded;
}

bool
SemanticMapper::computeCovariances(
  const std::vector<SemanticKeyframe::Ptr>& frames)
{
    // Need to ensure the graph is unfrozen to get accurate covariance
    // information, otherwise e.g. the first unfrozen frame will have a
    // near-zero covariance
    unfreezeAll();

    bool success;

    if (params_.covariance_backend == OptimizationBackend::CERES) {
        success = computeCovariancesWithCeres(frames);
    } else if (params_.covariance_backend == OptimizationBackend::GTSAM) {
        success = computeCovariancesWithGtsam(frames);
    } else if (params_.covariance_backend == OptimizationBackend::GTSAM_ISAM) {
        success = computeCovariancesWithGtsamIsam(frames);
    } else {
        throw std::runtime_error("Error: unsupported covariance backend.");
    }

    for (auto& frame : frames) {
        frame->covariance_computed_exactly() = true;
    }

    return success;
}

bool
SemanticMapper::computeLoopCovariances()
{
    // We're going to rely on the unfreezing of the loop via unfrozen_kfs_ to
    // determine which frames are in the loop We're only going to use a subset
    // of these frame covariances. Specifically we only need the covariance
    // information for frames which were the last frame in this loop to observe
    // an object...
    std::unordered_set<int> needed_covariance_frames;

    for (auto& kf : keyframes_) {
        if (unfrozen_kfs_.count(kf->index())) {

            for (auto& obj : kf->visible_objects()) {
                // Determine which was the last frame to observe it
                int max_frame = 0;
                for (auto& observing_frame : obj->keyframe_observations()) {
                    if (observing_frame->inGraph()) {
                        max_frame =
                          std::max(observing_frame->index(), max_frame);
                    }
                }
                needed_covariance_frames.insert(max_frame);
            }
        }
    }

    // assemble vector of actual frames
    std::vector<SemanticKeyframe::Ptr> frames;
    for (auto& idx : needed_covariance_frames) {
        frames.push_back(getKeyframeByIndex(idx));
    }

    return computeCovariances(frames);
}

bool
SemanticMapper::computeCovariancesWithCeres(
  const std::vector<SemanticKeyframe::Ptr>& frames)
{
    TIME_TIC;

    std::unique_lock<std::mutex> lock(graph_mutex_);

    for (auto frame : frames) {

        bool cov_succeeded =
          graph_->computeMarginalCovariance({ frame->graph_node() });

        if (!cov_succeeded) {
            ROS_WARN("Covariance computation failed!");
            return false;
        }

        // TODO how  best to handle this
        lock.unlock();
        std::unique_lock<std::mutex> map_lock(map_mutex_, std::defer_lock);
        std::lock(lock, map_lock);

        frame->covariance() =
          graph_->getMarginalCovariance({ frame->graph_node() });
    }

    ROS_INFO_STREAM("Covariance computation took " << TIME_TOC << " ms.");
    return true;
}

bool
SemanticMapper::computeCovariancesWithGtsam(
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
        getKeyframeByIndex(0)->key(),
        // getKeyframeByIndex(0)->pose()
        gtsam_values->at<gtsam::Pose3>(symbol_shorthand::X(0)));

    // Eigen::VectorXd prior_noise(6);
    // prior_noise << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    // auto gtsam_prior_noise =
    // gtsam::noiseModel::Diagonal::Sigmas(prior_noise); auto origin_factor =
    // util::allocate_aligned<gtsam::PriorFactor<gtsam::Pose3>>(
    //     getKeyframeByIndex(0)->key(),
    //     getKeyframeByIndex(0)->pose(),
    //     gtsam_prior_noise
    // );

    gtsam_graph->push_back(origin_factor);

    try {
        gtsam::Marginals marginals(*gtsam_graph, *gtsam_values);

        for (auto& frame : frames) {
            // ROS_INFO_STREAM("Computing covariance for " <<
            // DefaultKeyFormatter(frame->key()));
            auto cov = marginals.marginalCovariance(frame->key());

            std::lock_guard<std::mutex> map_lock(map_mutex_);

            frame->covariance() = cov;
        }

        ROS_INFO_STREAM("Covariance computation took " << TIME_TOC << " ms.");
    } catch (gtsam::IndeterminantLinearSystemException& e) {
        ROS_WARN_STREAM("Covariance computation failed! Error: " << e.what());

        // If this was detected near a semantic object, remove it from the
        // estimation
        Symbol bad_variable(e.nearbyVariable());

        std::lock_guard<std::mutex> lock(map_mutex_);

        if (bad_variable.chr() == 'c') {
            auto obj = getObjectByIndex(bad_variable.index());
            if (obj)
                obj->removeFromEstimation();
        } else if (bad_variable.chr() == 'l') {
            auto obj = getObjectByKeypointKey(bad_variable.key());
            if (obj)
                obj->removeFromEstimation();
        } else if (bad_variable.chr() == 'o') {
            auto obj = getObjectByKey(bad_variable.key());
            if (obj)
                obj->removeFromEstimation();
        }

        // and same for geometric landmarks
        if (bad_variable.chr() == 'g') {
            geom_handler_->removeLandmark(bad_variable.index());
        }

        return false;
    }
    // catch (std::out_of_range& e) {
    //     // ?? why does this happen??
    //     ROS_WARN_STREAM("Covariance computation failed! Error: " <<
    //     e.what()); return false;
    // }

    return true;
}

EstimatedObject::Ptr
SemanticMapper::getObjectByIndex(int index)
{
    // kind of a dumb method but keeping it for consistency
    return estimated_objects_[index];
}

EstimatedObject::Ptr
SemanticMapper::getObjectByKeypointKey(Key key)
{
    // Have to iterate over each object in this case
    for (auto& obj : estimated_objects_) {
        int64_t found_kp = obj->findKeypointByKey(key);
        if (found_kp >= 0) {
            return obj;
        }
    }

    return nullptr;
}

bool
SemanticMapper::computeCovariancesWithGtsamIsam(
  const std::vector<SemanticKeyframe::Ptr>& frames)
{
    bool created_isam_now = false;

    if (!isam_) {
        gtsam::ISAM2Params isam_params;
        isam_params.relinearizeThreshold = 0.5;

        isam_ = util::allocate_aligned<gtsam::ISAM2>(isam_params);
        created_isam_now = true;
    }

    TIME_TIC;

    std::lock_guard<std::mutex> lock(graph_mutex_);

    boost::shared_ptr<gtsam::NonlinearFactorGraph> gtsam_graph =
      graph_->getGtsamGraph();
    boost::shared_ptr<gtsam::Values> gtsam_values = graph_->getGtsamValues();

    gtsam::FactorIndices removed_factors = computeRemovedFactors(gtsam_graph);

    auto incremental_graph = computeIncrementalGraph(gtsam_graph);
    auto incremental_values = computeIncrementalValues(gtsam_values);

    // Anchor the origin
    if (created_isam_now) {
        isam_origin_factor_ =
          util::allocate_aligned<gtsam::NonlinearEquality<gtsam::Pose3>>(
            getKeyframeByIndex(0)->key(), getKeyframeByIndex(0)->pose());

        incremental_graph->push_back(isam_origin_factor_);
    }

    try {
        gtsam::ISAM2Result isam_result = isam_->update(
          *incremental_graph, *incremental_values, removed_factors);

        // update factor indices
        for (size_t i = 0; i < incremental_graph->size(); ++i) {
            isam_factor_indices_[incremental_graph->at(i).get()] =
              isam_result.newFactorsIndices[i];
        }

        for (auto& frame : frames) {
            auto cov = isam_->marginalCovariance(frame->key());

            std::lock_guard<std::mutex> map_lock(map_mutex_);

            frame->covariance() = cov;
        }

        ROS_INFO_STREAM("Covariance computation took " << TIME_TOC << " ms.");
    } catch (gtsam::IndeterminantLinearSystemException& e) {
        ROS_WARN("Covariance computation failed!");
        return false;
    }

    return true;
}

SemanticKeyframe::Ptr
SemanticMapper::getLastKeyframeInGraph()
{
    SemanticKeyframe::Ptr frame = nullptr;

    for (int i = keyframes_.size() - 1; i >= 0; --i) {
        if (keyframes_[i]->inGraph()) {
            frame = keyframes_[i];
            break;
        }
    }

    return frame;
}

gtsam::FactorIndices
SemanticMapper::computeRemovedFactors(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph)
{
    gtsam::FactorIndices removed;

    if (!factors_in_graph_)
        return removed;

    // Iterate over our factors and see aren't which in graph
    for (auto our_fac : *factors_in_graph_) {

        if (our_fac == isam_origin_factor_)
            continue;

        bool exists = false;
        for (auto fac : *graph) {
            if (our_fac == fac) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            removed.push_back(isam_factor_indices_[our_fac.get()]);

            // should probably remove the pair from isam_factor_indices_ too but
            // whatever
        }
    }

    return removed;
}

boost::shared_ptr<gtsam::Values>
SemanticMapper::computeIncrementalValues(
  boost::shared_ptr<gtsam::Values> values)
{
    // Compute a new values containing all *new* values
    if (!values_in_graph_) {
        values_in_graph_ = values;
        return values;
    }

    auto new_values = util::allocate_aligned<gtsam::Values>();

    for (auto key_value : *values) {
        if (values_in_graph_->find(key_value.key) == values_in_graph_->end()) {
            new_values->insert(key_value.key, key_value.value);
        }
    }

    values_in_graph_ = values;

    return new_values;
}

boost::shared_ptr<gtsam::NonlinearFactorGraph>
SemanticMapper::computeIncrementalGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph)
{
    if (!factors_in_graph_) {
        factors_in_graph_ = graph;
        return graph;
    }

    // this will be slow...
    auto new_graph = util::allocate_aligned<gtsam::NonlinearFactorGraph>();

    for (auto fac : *graph) {

        // compare to every existing factor
        bool exists = false;
        for (auto existing_fac : *factors_in_graph_) {
            if (fac == existing_fac) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            new_graph->push_back(fac);
        }
    }

    factors_in_graph_ = graph;

    return new_graph;
}

void
SemanticMapper::prepareGraphNodes()
{
    std::unique_lock<std::mutex> map_lock(map_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::lock(map_lock, graph_lock);

    for (auto& kf : keyframes_) {
        if (kf->inGraph()) {
            kf->graph_node()->pose() = kf->pose();
        }
    }

    for (auto& obj : estimated_objects_) {
        if (obj->inGraph()) {
            obj->prepareGraphNode();
        }
    }
}

void
SemanticMapper::commitGraphSolution()
{
    std::unique_lock<std::mutex> map_lock(map_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::lock(map_lock, graph_lock);

    // Find the latest keyframe in the graph optimization to check the computed
    // transformation
    SemanticKeyframe::Ptr last_in_graph = getLastKeyframeInGraph();

    // for (auto& kf : keyframes_) {
    //     std::cout << kf->graph_node()->pose().rotation().coeffs().norm() <<
    //     std::endl;
    // }

    const Pose3& old_pose = last_in_graph->pose();

    if (params_.optimization_backend == OptimizationBackend::CERES) {
        Pose3 new_pose = last_in_graph->graph_node()->pose();
        new_pose.rotation().normalize();

        Pose3 old_T_new = old_pose.inverse() * new_pose;
        old_T_new.rotation().normalize();

        // Pose3 new_map_T_old_map = new_pose * old_pose.inverse();
        // new_map_T_old_map.rotation().normalize();

        for (auto& kf : keyframes_) {
            if (kf->inGraph()) {
                kf->pose() = kf->graph_node()->pose();
                kf->pose().rotation().normalize();
            } else {
                // Keyframes not yet in the graph will be later so just
                // propagate the computed transform forward
                kf->pose() = kf->pose() * old_T_new;

                // kf->pose() = new_map_T_old_map * kf->pose();
            }
        }

        for (auto& obj : estimated_objects_) {
            if (obj->inGraph()) {
                obj->commitGraphSolution();
            } else if (!obj->bad()) {
                // Update the object based on recomputed camera poses
                // ROS_ERROR_STREAM("OBJ not in graph");
                obj->optimizeStructure();
            }
        }

    } else {
        // GTSAM
        Pose3 new_pose = gtsam_values_.at<gtsam::Pose3>(last_in_graph->key());

        Pose3 new_map_T_old_map = new_pose * old_pose.inverse();

        for (auto& kf : keyframes_) {
            if (kf->inGraph()) {
                kf->pose() = gtsam_values_.at<gtsam::Pose3>(kf->key());
            } else {
                kf->pose() = new_map_T_old_map * kf->pose();
            }
        }

        for (auto& obj : estimated_objects_) {
            if (obj->inGraph()) {
                obj->commitGtsamSolution(gtsam_values_);
            } else if (!obj->bad()) {
                obj->optimizeStructure();
            }
        }
    }
}

bool
SemanticMapper::tryOptimize()
{
    if (!graph_->modified())
        return false;

    TIME_TIC;

    invalidate_local_optimization_ = false;

    prepareGraphNodes();

    bool solve_succeeded = solveGraph();

    if (solve_succeeded && !invalidate_local_optimization_) {
        commitGraphSolution();
        ROS_INFO_STREAM(
          fmt::format("Solved {} nodes and {} edges in {:.2f} ms.",
                      graph_->num_nodes(),
                      graph_->num_factors(),
                      TIME_TOC));
        last_optimized_kf_index_ = getLastKeyframeInGraph()->index();
        return true;
    } else if (!solve_succeeded) {
        ROS_INFO_STREAM("Graph solve failed");
    }

    // No matter what we want this invalidation to be false here. It is only
    // meant to check for loop closures that happen between calls to
    // solveGraph() and commitGraphSolution().
    invalidate_local_optimization_ = false;
    return false;
}

bool
SemanticMapper::optimizeFully()
{
    TIME_TIC;

    // The only difference between this function and tryOptimize is that
    // we're not going to limit the amount of time the solver takes (too much??)

    prepareGraphNodes();

    auto old_options = graph_->solver_options();

    graph_->solver_options().max_solver_time_in_seconds = 10;
    graph_->solver_options().linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    // graph_->solver_options().linear_solver_type = ceres::ITERATIVE_SCHUR;
    // graph_->solver_options().minimizer_type = ceres::LINE_SEARCH;

    // graph_->setSolverOptions(full_options);

    // bool solve_succeeded = solveGraph();

    bool solve_succeeded;
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        solve_succeeded = graph_->solve(true);
    }

    // Set the solver back to normal
    graph_->setSolverOptions(old_options);

    if (solve_succeeded) {
        commitGraphSolution();
        ROS_INFO_STREAM(
          fmt::format("Solved {} nodes and {} edges in {:.2f} ms.",
                      graph_->num_nodes(),
                      graph_->num_factors(),
                      TIME_TOC));
        last_optimized_kf_index_ = getLastKeyframeInGraph()->index();
        return true;
    } else {
        ROS_INFO_STREAM("Graph solve failed");
        return false;
    }
}

bool
SemanticMapper::optimizeEssential()
{
    TIME_TIC;

    prepareGraphNodes();

    // auto essential_options = solver_options_;
    essential_graph_->solver_options() = graph_->solver_options();

    essential_graph_->solver_options().max_solver_time_in_seconds = 1;
    essential_graph_->solver_options().max_num_iterations = 100000;

    // essential_graph_->solver_options().linear_solver_type =
    // ceres::ITERATIVE_SCHUR;

    // essential_graph_->solver_options().minimizer_type = ceres::LINE_SEARCH;
    // essential_graph_->solver_options().line_search_direction_type =
    // ceres::NONLINEAR_CONJUGATE_GRADIENT;

    // essential_graph_->solver_options().linear_solver_type =
    // ceres::SPARSE_SCHUR;

    // essential_graph_->setSolverOptions(essential_options);

    bool solve_succeeded;

    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        solve_succeeded = essential_graph_->solve(true);
    }

    if (solve_succeeded) {
        commitGraphSolution();
        ROS_INFO_STREAM(fmt::format(
          "Solved ESSENTIAL graph with {} nodes and {} edges in {:.2f} ms.",
          essential_graph_->num_nodes(),
          essential_graph_->num_factors(),
          TIME_TOC));
    } else {
        ROS_INFO_STREAM("ESSENTIAL Graph solve failed");
    }

    return solve_succeeded;
}

bool
SemanticMapper::solveGraph()
{
    std::lock_guard<std::mutex> lock(graph_mutex_);

    if (params_.optimization_backend == OptimizationBackend::CERES) {
        return graph_->solve(false);
    } else {
        auto gtsam_graph = graph_->getGtsamGraph();
        auto gtsam_values = graph_->getGtsamValues();

        // Have to anchor the origin here ourselves
        Eigen::VectorXd prior_noise(6);
        prior_noise << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        auto gtsam_prior_noise =
          gtsam::noiseModel::Diagonal::Sigmas(prior_noise);
        auto origin_factor =
          util::allocate_aligned<gtsam::PriorFactor<gtsam::Pose3>>(
            getKeyframeByIndex(0)->key(),
            getKeyframeByIndex(0)->pose(),
            gtsam_prior_noise);

        gtsam_graph->push_back(origin_factor);

        gtsam::LevenbergMarquardtParams lm_params;
        lm_params.orderingType = gtsam::Ordering::OrderingType::METIS;
        lm_params.setVerbosityLM("SUMMARY");
        // lm_params.setVerbosity("ERROR");
        // lm_params.print("LM PARAMS");
        gtsam::LevenbergMarquardtOptimizer optimizer(
          *gtsam_graph, *gtsam_values, lm_params);

        // optimizer.params().ordering->print("ordering");

        // gtsam_graph->print("graph:");

        // gtsam::LevenbergMarquardtOptimizer optimizer(*gtsam_graph,
        // *gtsam_values);

        try {
            gtsam_values_ = optimizer.optimize();
            return true;
        } catch (gtsam::IndeterminantLinearSystemException& e) {
            return false;
        }
    }
}

void
SemanticMapper::loadModelFiles(std::string path)
{
    boost::filesystem::path model_path(path);
    for (auto i = boost::filesystem::directory_iterator(model_path);
         i != boost::filesystem::directory_iterator();
         i++) {
        if (!boost::filesystem::is_directory(i->path())) {
            ROS_INFO("Loading %s", i->path().filename().string().c_str());
            std::string file_name = i->path().filename().string();
            std::string class_name = file_name.substr(0, file_name.find('.'));

            object_models_[class_name] =
              geometry::readModelFile(i->path().string());
        } else {
            continue;
        }
    }
    ROS_INFO("Loaded %d models", (int)(object_models_.size()));
}

bool
SemanticMapper::loadCalibration()
{
    // Read from the ros parameter server...

    double fx, fy, s, u0, v0, k1, k2, p1, p2;
    s = 0.0; // skew
    if (!pnh_.getParam("cam_cx", u0) || !pnh_.getParam("cam_cy", v0) ||
        !pnh_.getParam("cam_d0", k1) || !pnh_.getParam("cam_d1", k2) ||
        !pnh_.getParam("cam_d2", p1) || !pnh_.getParam("cam_d3", p2) ||
        !pnh_.getParam("cam_fx", fx) || !pnh_.getParam("cam_fy", fy)) {
        ROS_ERROR("Error: unable to read camera intrinsic parameters.");
        return false;
    }

    camera_calibration_ =
      boost::make_shared<CameraCalibration>(fx, fy, s, u0, v0, k1, k2, p1, p2);

    std::vector<double> I_p_C, I_q_C;

    if (!getRosParam(pnh_, "I_p_C", I_p_C) ||
        !getRosParam(pnh_, "I_q_C", I_q_C)) {
        ROS_ERROR("Error: unable to read camera extrinsic parameters.");
        return false;
    }

    Eigen::Quaterniond q;
    Eigen::Vector3d p;

    q.x() = I_q_C[0];
    q.y() = I_q_C[1];
    q.z() = I_q_C[2];
    q.w() = I_q_C[3];
    q.normalize();

    p << I_p_C[0], I_p_C[1], I_p_C[2];

    I_T_C_ = Pose3(q, p);

    return true;
}

bool
SemanticMapper::loadParameters()
{
    std::string optimization_backend;
    std::string covariance_backend;

    if (!pnh_.getParam("keypoint_activation_threshold",
                       params_.keypoint_activation_threshold) ||

        !pnh_.getParam("min_object_n_keypoints",
                       params_.min_object_n_keypoints) ||
        !pnh_.getParam("min_landmark_observations",
                       params_.min_landmark_observations) ||

        !pnh_.getParam("structure_regularization_factor",
                       params_.structure_regularization_factor) ||
        !pnh_.getParam("robust_estimator_parameter",
                       params_.robust_estimator_parameter) ||

        !pnh_.getParam("structure_error_coefficient",
                       params_.structure_error_coefficient) ||
        !pnh_.getParam("include_objects_in_graph",
                       params_.include_objects_in_graph) ||

        !pnh_.getParam("new_landmark_weight_threshold",
                       params_.new_landmark_weight_threshold) ||
        !pnh_.getParam("mahal_thresh_assign", params_.mahal_thresh_assign) ||
        !pnh_.getParam("mahal_thresh_init", params_.mahal_thresh_init) ||
        !pnh_.getParam("keypoint_initialization_depth_sigma",
                       params_.keypoint_initialization_depth_sigma) ||
        !pnh_.getParam("constraint_weight_threshold",
                       params_.constraint_weight_threshold) ||
        !pnh_.getParam("keypoint_msmt_sigma", params_.keypoint_msmt_sigma) ||
        !pnh_.getParam("min_observed_keypoints_to_initialize",
                       params_.min_observed_keypoints_to_initialize) ||
        !pnh_.getParam("keyframe_translation_threshold",
                       params_.keyframe_translation_threshold) ||
        !pnh_.getParam("keyframe_rotation_threshold",
                       params_.keyframe_rotation_threshold) ||
        !pnh_.getParam(
          "keyframe_translation_without_measurement_threshold",
          params_.keyframe_translation_without_measurement_threshold) ||
        !pnh_.getParam(
          "keyframe_rotation_without_measurement_threshold",
          params_.keyframe_rotation_without_measurement_threshold) ||
        !pnh_.getParam("optimization_backend", optimization_backend) ||
        !pnh_.getParam("covariance_backend", covariance_backend) ||
        !pnh_.getParam("include_geometric_features",
                       include_geometric_features_) ||
        !pnh_.getParam("verbose_optimization", verbose_optimization_) ||
        !pnh_.getParam("covariance_delay", covariance_delay_) ||
        !pnh_.getParam("max_optimization_time", max_optimization_time_) ||
        !pnh_.getParam("loop_closure_threshold", loop_closure_threshold_) ||
        !pnh_.getParam("smoothing_length", smoothing_length_) ||
        !pnh_.getParam("use_manual_elimination_ordering",
                       params_.use_manual_elimination_ordering)) {

        ROS_ERROR("Unable to load object handler parameters");
        return false;
    }

    if (covariance_backend == "GTSAM") {
        params_.covariance_backend = OptimizationBackend::GTSAM;
    } else if (covariance_backend == "CERES") {
        params_.covariance_backend = OptimizationBackend::CERES;
    } else if (covariance_backend == "GTSAM_ISAM") {
        params_.covariance_backend = OptimizationBackend::GTSAM_ISAM;
    } else {
        throw std::runtime_error("Unsupported covariance backend");
    }

    if (optimization_backend == "GTSAM") {
        params_.optimization_backend = OptimizationBackend::GTSAM;
    } else if (optimization_backend == "CERES") {
        params_.optimization_backend = OptimizationBackend::CERES;
    } else {
        throw std::runtime_error("Unsupported optimization backend");
    }

    return true;
}

bool
SemanticMapper::keepFrame(
  const object_pose_interface_msgs::KeypointDetections& msg)
{
    if (keyframes_.empty())
        return true;

    // if (msg.detections.size() > 0) return true;

    auto last_keyframe = keyframes_.back();

    Pose3 relpose;
    bool got_relpose = odometry_handler_->getRelativePoseEstimate(
      last_keyframe->time(), msg.header.stamp, relpose);

    if (!got_relpose) {
        // ROS_WARN_STREAM("Too few odometry messages received to get keyframe
        // relative pose");
        return true;
    }

    if (msg.detections.size() > 0) {
        if (relpose.translation().norm() >
              params_.keyframe_translation_threshold ||
            2 * std::acos(relpose.rotation().w()) * 180 / 3.14159 >
              params_.keyframe_rotation_threshold) {
            return true;
        } else {
            return false;
        }
    } else {
        if (relpose.translation().norm() >
              params_.keyframe_translation_without_measurement_threshold ||
            2 * std::acos(relpose.rotation().w()) * 180 / 3.14159 >
              params_.keyframe_rotation_without_measurement_threshold) {
            return true;
        } else {
            return false;
        }
    }
}

void
SemanticMapper::anchorOrigin()
{
    SemanticKeyframe::Ptr origin_kf =
      odometry_handler_->originKeyframe(ros::Time(0));

    origin_kf->addToGraph(graph_);
    graph_->setNodeConstant(origin_kf->graph_node());

    origin_kf->addToGraph(essential_graph_);
    essential_graph_->setNodeConstant(origin_kf->graph_node());

    keyframes_.push_back(origin_kf);
}

bool
SemanticMapper::haveNextKeyframe()
{
    // Check if we have a candidate keyframe waiting in our message queue
    // This will modify the message queue in that it discards non-keyframes,
    // but it will not remove the keyframe from the queue

    std::lock_guard<std::mutex> lock(queue_mutex_);

    while (!msg_queue_.empty()) {
        const auto& msg = msg_queue_.front();

        if (keepFrame(msg)) {
            return true;
        } else {
            msg_queue_.pop_front();
        }
    }

    return false;
}

SemanticKeyframe::Ptr
SemanticMapper::tryFetchNextKeyframe()
{
    object_pose_interface_msgs::KeypointDetections msg;
    SemanticKeyframe::Ptr next_keyframe = nullptr;

    {
        std::unique_lock<std::mutex> lock(queue_mutex_, std::defer_lock);
        std::unique_lock<std::mutex> map_lock(map_mutex_, std::defer_lock);
        std::lock(lock, map_lock);

        while (!msg_queue_.empty()) {
            msg = msg_queue_.front();

            if (keepFrame(msg)) {
                next_keyframe =
                  odometry_handler_->createKeyframe(msg.header.stamp);
                if (next_keyframe) {
                    msg_queue_.pop_front();
                    next_keyframe->measurements() =
                      processObjectDetectionMessage(msg, next_keyframe->key());

                    // Copy the most recent covariance into this frame for now
                    // as an OK approximation, it will be updated later if we
                    // have the processing bandwidth.
                    next_keyframe->covariance() = last_kf_covariance_;
                    keyframes_.push_back(next_keyframe);
                    break;
                } else {
                    // Want to keep this keyframe but odometry handler can't
                    // make it for us yet -- try again later
                    return nullptr;
                }
            }

            msg_queue_.pop_front();
        }
    }

    return next_keyframe;
}

bool
SemanticMapper::updateKeyframeObjects(SemanticKeyframe::Ptr frame)
{
    if (!frame)
        return false;

    std::lock_guard<std::mutex> lock(map_mutex_);

    // TODO fix this bad approximation
    frame->covariance() = last_kf_covariance_;

    if (frame->measurements().size() > 0) {

        // Create the list of measurements we need to associate.
        // Identify which measurements have been tracked from already known
        // objects
        std::vector<size_t> measurement_index;
        std::map<size_t, size_t> known_das;

        for (size_t i = 0; i < frame->measurements().size(); ++i) {
            auto known_id_it =
              object_track_ids_.find(frame->measurements()[i].track_id);
            if (known_id_it == object_track_ids_.end()) {
                // track not known, perform actual data association

                // If data association is not known, we need to perform actual
                // association at the object level. to do this we require that a
                // certain number of keypoints were actually observed so we
                // don't make spurious associations
                // --> objects not tracked and with too few kps observed are
                // effectively thrown away
                if (frame->measurements()[i].n_keypoints_observed >= 4) {
                    measurement_index.push_back(i);
                }
            } else {
                // track ID known, take data association as given
                known_das[i] = known_id_it->second;
            }
        }

        // Create the list of objects we need to associate.
        // count visible landmarks & create mapping from list of visible to list
        // of all
        std::vector<bool> visible = predictVisibleObjects(frame);
        size_t n_visible = 0;
        std::vector<size_t> object_index;
        for (size_t j = 0; j < estimated_objects_.size(); ++j) {
            if (visible[j]) {
                n_visible++;
                estimated_objects_[j]->setIsVisible(frame);
                object_index.push_back(j);
            }
        }

        Eigen::MatrixXd mahals =
          Eigen::MatrixXd::Zero(measurement_index.size(), n_visible);

        for (size_t i = 0; i < measurement_index.size(); ++i) {
            for (size_t j = 0; j < n_visible; ++j) {
                // If we're currently waiting for a loop closure to be
                // processed, we don't want to add measurements to old
                // objects right now. they will be processed afterwards.
                // instead, we will continue to create a new local map...
                // Do this by inflating mahal distances to them
                // ^- wait why this is dumb

                // if (operation_mode_ == OperationMode::NORMAL ||
                //     static_cast<int>(
                //   estimated_objects_[object_index[j]]->last_seen()) >
                //   frame->index() - loop_closure_threshold_) {
                mahals(i, j) = estimated_objects_[object_index[j]]
                                 ->computeMahalanobisDistance(
                                   frame->measurements()[measurement_index[i]]);
                // } else {
                //     mahals(i, j) = 1e5;
                // }
            }
        }

        // std::cout << "Mahals:\n" << mahals << std::endl;

        // std::cout << "Mahals: ";
        // for (int i = 0; i < measurement_index.size(); ++i) {
        //     std::cout << "MSMT " << measurement_index[i] << ": ";

        //     for (int j = 0; j < n_visible; ++j) {
        //         std::cout << DefaultKeyFormatter(sym::O(
        //                        estimated_objects_[object_index[j]]->id()))
        //                   << ": " << mahals(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;

        Eigen::MatrixXd weights_matrix =
          MLDataAssociator(params_).computeConstraintWeights(mahals);

        addMeasurementsToObjects(frame,
                                 frame->measurements(),
                                 measurement_index,
                                 known_das,
                                 weights_matrix,
                                 object_index);

        // calling the "update" method on each of our objects allows them to
        // remove themselves from the estimation if they're poorly localized
        // and haven't been observed recently. These objects can cause poor
        // data association and errors down the road otherwise
        for (auto& obj : estimated_objects_) {
            if (!obj->bad()) {
                obj->update(frame);
            }
        }
    }

    return true;
}

aligned_vector<ObjectMeasurement>
SemanticMapper::processObjectDetectionMessage(
  const object_pose_interface_msgs::KeypointDetections& msg,
  Key keyframe_key)
{
    aligned_vector<ObjectMeasurement> object_measurements;

    for (auto detection : msg.detections) {
        auto model_it = object_models_.find(detection.obj_name);
        if (model_it == object_models_.end()) {
            ROS_WARN_STREAM("No object model for class " << detection.obj_name);
            continue;
        }

        geometry::ObjectModelBasis model = model_it->second;

        // Build vectors etc to optimize structure
        Eigen::MatrixXd normalized_coords(3, detection.x.size());
        Eigen::VectorXd weights(detection.x.size());

        for (size_t i = 0; i < detection.x.size(); ++i) {
            Eigen::Vector2d img_coords(detection.x[i], detection.y[i]);
            normalized_coords.block<2, 1>(0, i) =
              camera_calibration_->calibrate(img_coords);
            normalized_coords(2, i) = 1.0;

            weights(i) = detection.probabilities[i];
        }

        geometry::StructureResult result;

        try {
            result = geometry::optimizeStructureFromProjection(
              normalized_coords,
              model,
              weights,
              false /* compute depth covariance? */);

            // ROS_WARN_STREAM("Optimized structure.");
            // std::cout << "t = " << result.t.transpose() << std::endl;
        } catch (std::exception& e) {
            ROS_WARN_STREAM("Structure optimization failed:\n" << e.what());
            continue;
        }

        // Build object measurement structure
        ObjectMeasurement obj_msmt;
        obj_msmt.observed_key = keyframe_key;
        obj_msmt.stamp = detection.header.stamp;
        obj_msmt.platform = "zed";
        obj_msmt.frame = detection.header.frame_id;
        obj_msmt.global_msmt_id = measurements_processed_++;
        obj_msmt.obj_name = detection.obj_name;
        obj_msmt.track_id = detection.bounding_box.id;
        obj_msmt.t = result.t;
        obj_msmt.q = Eigen::Quaterniond(result.R);

        obj_msmt.bbox.xmin = detection.bounding_box.xmin;
        obj_msmt.bbox.ymin = detection.bounding_box.ymin;
        obj_msmt.bbox.xmax = detection.bounding_box.xmax;
        obj_msmt.bbox.ymax = detection.bounding_box.ymax;

        double bbox_dim = 0.5 * (obj_msmt.bbox.xmax - obj_msmt.bbox.xmin +
                                 obj_msmt.bbox.ymax - obj_msmt.bbox.ymin);

        size_t n_keypoints_observed = 0;

        for (size_t i = 0; i < detection.x.size(); ++i) {
            KeypointMeasurement kp_msmt;

            kp_msmt.measured_key = keyframe_key;
            kp_msmt.stamp = detection.header.stamp;
            kp_msmt.platform = "zed";
            kp_msmt.obj_name = detection.obj_name;

            kp_msmt.pixel_measurement << detection.x[i], detection.y[i];
            kp_msmt.normalized_measurement =
              normalized_coords.block<2, 1>(0, i);

            kp_msmt.score = detection.probabilities[i];
            kp_msmt.depth = result.Z(i);

            // covariance estimation in the structure optimization may have
            // failed
            if (result.Z_covariance.size() > 0) {
                kp_msmt.depth_sigma = std::sqrt(result.Z_covariance(i));
            } else {
                kp_msmt.depth_sigma =
                  params_.keypoint_initialization_depth_sigma;
            }

            kp_msmt.kp_class_id = i;

            // TODO what here?
            kp_msmt.pixel_sigma =
              std::max(0.1 * bbox_dim, params_.keypoint_msmt_sigma);
            // kp_msmt.pixel_sigma = params_.keypoint_msmt_sigma;

            // ROS_INFO_STREAM("px sigma = " << kp_msmt.pixel_sigma);

            if (kp_msmt.score > params_.keypoint_activation_threshold) {
                kp_msmt.observed = true;
                n_keypoints_observed++;
            } else {
                kp_msmt.observed = false;
            }

            obj_msmt.keypoint_measurements.push_back(kp_msmt);
        }

        if (n_keypoints_observed > 0) {
            obj_msmt.n_keypoints_observed = n_keypoints_observed;
            object_measurements.push_back(obj_msmt);
        }
    }

    return object_measurements;
}

EstimatedObject::Ptr
SemanticMapper::getObjectByKey(Key key)
{
    return estimated_objects_[Symbol(key).index()];
}

Eigen::MatrixXd
SemanticMapper::getPlx(Key key1, Key key2)
{
    // First get the covariances according to the *local* optimization
    auto obj = getObjectByKey(key1);
    Eigen::MatrixXd Plx_local = obj->getPlx(key1, key2);

    // Find the most recent camera pose that observed this object
    int max_index = -1;
    for (const auto& frame : obj->keyframe_observations()) {
        max_index = std::max(frame->index(), max_index);
    }

    auto kf_old = getKeyframeByIndex(max_index);
    auto kf = getKeyframeByKey(key2);

    // TODO REMOVE THIS REMOVE THIS
    // if (kf_old->inGraph() && !kf_old->covariance_computed_exactly())
    // computeCovariances({kf_old}); if (kf->inGraph()) {
    //     if (!kf->covariance_computed_exactly()) computeCovariances({kf});
    // } else {
    //     auto last_kf = getLastKeyframeInGraph();
    //     if (!last_kf->covariance_computed_exactly())
    //     computeCovariances({last_kf}); kf->covariance() =
    //     last_kf->covariance();
    // }

    // Plx local now is the covariance of the landmarks *expressed in the
    // latest robot frame* What we really want is the covariance of the
    // estimates *in the map/global frame* Need to use the Jacobian of the
    // pose transformation Let {G} denote the global frame, {L} denote the
    // frame of the local robot frame
    Pose3 G_T_L = kf_old->pose();

    size_t H_size = 3 * obj->keypoints().size();

    Eigen::MatrixXd D_GL_DLl_full = Eigen::MatrixXd::Zero(H_size, H_size);
    Eigen::MatrixXd D_Gl_Dx_full = Eigen::MatrixXd::Zero(H_size, 6);

    for (size_t i = 0; i < obj->keypoints().size(); ++i) {
        Eigen::MatrixXd D_Gl_DLl; // jacobian of global point w.r.t. local point
        Eigen::MatrixXd
          D_Gl_Dx; // jacobian of global point w.r.t. local robot pose

        // We actually estimated the keypoint in the local frame but only
        // expose it pre-transformed into the global frame, whatever
        Eigen::Vector3d L_l =
          G_T_L.transform_to(obj->keypoints()[i]->position());

        // Eigen::Vector3d G_l =
        G_T_L.transform_from(L_l, D_Gl_Dx, D_Gl_DLl);

        // Now we have the same dumb thing where D_gl_Dq is in the ambient
        // space but our covariances are in the tangent space
        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> Hquat_space;
        QuaternionLocalParameterization().ComputeJacobian(G_T_L.rotation_data(),
                                                          Hquat_space.data());

        D_GL_DLl_full.block<3, 3>(3 * i, 3 * i) = D_Gl_DLl;

        D_Gl_Dx_full.block<3, 3>(3 * i, 0) =
          D_Gl_Dx.block<3, 4>(0, 0) * Hquat_space;
        D_Gl_Dx_full.block<3, 3>(3 * i, 3) = D_Gl_Dx.block<3, 3>(0, 4);
    }

    // OK now we can finally compute the global covariance estimate
    size_t Plx_dim = 6 + 3 * obj->keypoints().size();

    Eigen::MatrixXd H_full = Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);
    H_full.topLeftCorner(H_size, H_size) = D_GL_DLl_full;
    H_full.topRightCorner(H_size, 6) = D_Gl_Dx_full;
    // H_full.bottomLeftCorner is Dx / Dlandmarks = 0
    H_full.bottomRightCorner(6, 6) = Eigen::MatrixXd::Identity(6, 6);

    Eigen::MatrixXd P_full = Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);
    P_full.topLeftCorner(H_size, H_size) =
      Plx_local.topLeftCorner(H_size, H_size);
    P_full.bottomRightCorner(6, 6) = kf_old->covariance();

    Eigen::MatrixXd global_covs = H_full * P_full * H_full.transpose();

    // Now the thing is we don't want Plx with x as the most recent keyframe
    // involved in its local optimization, we want Plx where x is the
    // current keyframe that we're performing data association in.
    //
    // All of this would be much simpler if we could just compute the actual
    // covariances between the graph keyframes and landmarks, we have enough
    // information in the factor graph to do so, but it's way too
    // computationally expensive
    //
    // We're doing a weird trick here...
    // Assume that the covariance between keyframes can be approximated by a
    // linear update, i.e. P2 = H*P1*H' for some H. We don't have access to
    // this H, but we have P1 and P2. Take Cholesky and write them as L1*L1'
    // and L2*L2', so L2*L2' = H*L1*L1'*H'. So we see L2 = H*L1. We further
    // have the constraint that x2 = H*x1.

    // auto kf_old = getKeyframeByIndex(Plxs_index_);

    if (kf_old->index() == kf->index()) {
        return global_covs;
    }

    // ROS_INFO_STREAM("Difference between this keyframe and last observed:
    // " << kf->index() - max_index);

    const Eigen::MatrixXd& Px1 = kf_old->covariance();
    const Eigen::MatrixXd& Px2 = kf->covariance();

    // Cholesky
    /*
    Eigen::LLT<Eigen::MatrixXd> llt1(Px1);
    Eigen::LLT<Eigen::MatrixXd> llt2(Px2);

    Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Lower> L1 =
    llt1.matrixL(); Eigen::MatrixXd L2 = llt2.matrixL();
    // H12.bottomRightCorner<6,6>() = llt2.matrixL() * L1.inverse();

    H12.bottomRightCorner<6,6>() = L1.solve<Eigen::OnTheRight>(L2);
    */

    // Eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp1(Px1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(Px2);

    Eigen::MatrixXd L1 = decomp1.operatorSqrt();
    Eigen::MatrixXd L2 = decomp2.operatorSqrt();

    // Build system to solve, [L2 x2] = H * [L1 x1] -> LHS = H*RHS -> LHS^T
    // = RHS^T H^T
    Eigen::MatrixXd LHS(L2.rows(), L2.cols() + 1);
    Eigen::MatrixXd RHS(L1.rows(), L1.cols() + 1);

    LHS.leftCols(L2.cols()) = L2;
    RHS.leftCols(L1.cols()) = L1;
    // This is wonky because the covariance is in the tangent space so the
    // pose is 6-dimensional but the actual state is 7. Ignore the 4th
    // quaternion component?? If the first three are equal the fourth will
    // be by the norm constraint
    Eigen::VectorXd x1_reduced(6), x2_reduced(6);
    x1_reduced.head<3>() = kf_old->pose().rotation().coeffs().head<3>();
    x1_reduced.tail<3>() = kf_old->pose().translation();
    x2_reduced.head<3>() = kf->pose().rotation().coeffs().head<3>();
    x2_reduced.tail<3>() = kf->pose().translation();

    LHS.rightCols<1>() = x2_reduced;
    RHS.rightCols<1>() = x1_reduced;

    // Solve the system to find our desired H
    Eigen::MatrixXd H12 = Eigen::MatrixXd::Identity(Plx_dim, Plx_dim);
    // H12.bottomRightCorner<6,6>() =
    // RHS.transpose().colPivHouseholderQr().solve(LHS.transpose()).transpose();

    H12.bottomRightCorner<6, 6>() =
      L1.transpose().ldlt().solve(L2.transpose()).transpose();

    Eigen::MatrixXd updated_covs = H12 * global_covs * H12.transpose();

    // for debugging just check cov with one landmark
    // Eigen::MatrixXd Plx_local_one(9,9);
    // Eigen::MatrixXd global_covs_one(9,9);
    // Eigen::MatrixXd H_one(9,9);
    // Eigen::MatrixXd updated_Plx_one(9,9);
    // // Eigen::MatrixXd true_Plx_one(9,9);

    // // Eigen::MatrixXd true_Plx = computePlxExact(obj->key(), key2);

    // // true_Plx_one.topLeftCorner<3,3>() = true_Plx.topLeftCorner<3,3>();
    // // true_Plx_one.topRightCorner<3,6>() =
    // true_Plx.topRightCorner<3,6>();
    // // true_Plx_one.bottomLeftCorner<6,3>() =
    // true_Plx.bottomLeftCorner<6,3>();
    // // true_Plx_one.bottomRightCorner<6,6>() =
    // true_Plx.bottomRightCorner<6,6>();

    // Plx_local_one.topLeftCorner<3,3>() = Plx_local.topLeftCorner<3,3>();
    // Plx_local_one.topRightCorner<3,6>() =
    // Plx_local.topRightCorner<3,6>();
    // Plx_local_one.bottomLeftCorner<6,3>() =
    // Plx_local.bottomLeftCorner<6,3>();
    // Plx_local_one.bottomRightCorner<6,6>() =
    // Plx_local.bottomRightCorner<6,6>();

    // global_covs_one.topLeftCorner<3,3>() =
    // global_covs.topLeftCorner<3,3>();
    // global_covs_one.topRightCorner<3,6>() =
    // global_covs.topRightCorner<3,6>();
    // global_covs_one.bottomLeftCorner<6,3>() =
    // global_covs.bottomLeftCorner<6,3>();
    // global_covs_one.bottomRightCorner<6,6>() =
    // global_covs.bottomRightCorner<6,6>();

    // H_one.topLeftCorner<3,3>() = H12.topLeftCorner<3,3>();
    // H_one.topRightCorner<3,6>() = H12.topRightCorner<3,6>();
    // H_one.bottomLeftCorner<6,3>() = H12.bottomLeftCorner<6,3>();
    // H_one.bottomRightCorner<6,6>() = H12.bottomRightCorner<6,6>();

    // updated_Plx_one.topLeftCorner<3,3>() =
    // updated_covs.topLeftCorner<3,3>();
    // updated_Plx_one.topRightCorner<3,6>() =
    // updated_covs.topRightCorner<3,6>();
    // updated_Plx_one.bottomLeftCorner<6,3>() =
    // updated_covs.bottomLeftCorner<6,3>();
    // updated_Plx_one.bottomRightCorner<6,6>() =
    // updated_covs.bottomRightCorner<6,6>();

    // std::cout << "OBJECT ID = " << obj->id() << std::endl;
    // std::cout << "Plx LOCAL: \n" << Plx_local_one << std::endl;
    // std::cout << "OLD FRAME covariance: \n" << kf_old->covariance() <<
    // std::endl; std::cout << "NEW FRAME covariance: \n" <<
    // kf->covariance() << std::endl; std::cout << "Plx GLOBAL: \n" <<
    // global_covs_one << std::endl; std::cout << "H: \n" << H_one <<
    // std::endl; std::cout << "Plx TRANSFORMED: \n" << updated_Plx_one <<
    // std::endl;
    // // std::cout << "TRUE Plx: \n" << true_Plx_one << std::endl;

    // if (!updated_covs.allFinite()) {

    //     std::cout << "H_full:\n" << H_full << std::endl;
    //     std::cout << "P_full:\n" << P_full << std::endl;

    //     std::cout << "Plx LOCAL: \n" << Plx_local << std::endl;
    //     std::cout << "Plx GLOBAL: \n" << global_covs << std::endl;
    //     std::cout << "H: \n" << H12 << std::endl;
    //     std::cout << "Plx TRANSFORMED: \n" << updated_covs << std::endl;

    // }

    return updated_covs;
}

Eigen::MatrixXd
SemanticMapper::computePlxExact(Key obj_key, Key x_key)
{
    boost::shared_ptr<gtsam::NonlinearFactorGraph> gtsam_graph;
    boost::shared_ptr<gtsam::Values> gtsam_values;

    {
        std::lock_guard<std::mutex> lock(graph_mutex_);

        gtsam_graph = graph_->getGtsamGraph();
        gtsam_values = graph_->getGtsamValues();
    }

    auto origin_factor =
      util::allocate_aligned<gtsam::NonlinearEquality<gtsam::Pose3>>(
        getKeyframeByIndex(0)->key(),
        gtsam_values->at<gtsam::Pose3>(symbol_shorthand::X(0)));

    gtsam_graph->push_back(origin_factor);

    auto obj = getObjectByKey(obj_key);
    auto kf = getKeyframeByKey(x_key);

    int Plx_dim = 3 * obj->keypoints().size() + 6;

    // If the obejct isn't in the graph we can't do this...
    if (!obj->inGraph()) {
        return Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);
    }

    // Actually we should ignore x_key and get the latest keyframe in the
    // graph
    x_key = getLastKeyframeInGraph()->key();

    gtsam::KeyVector keys;
    for (size_t i = 0; i < obj->keypoints().size(); ++i) {
        keys.push_back(obj->keypoints()[i]->key());
    }
    keys.push_back(x_key);

    try {
        gtsam::Marginals marginals(*gtsam_graph, *gtsam_values);

        auto joint_cov = marginals.jointMarginalCovariance(keys);

        return joint_cov.fullMatrix();

    } catch (gtsam::IndeterminantLinearSystemException& e) {
        ROS_WARN_STREAM("Covariance computation failed! Error: " << e.what());

        return Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);
    }
}

bool
SemanticMapper::createNewObject(const ObjectMeasurement& measurement,
                                const Pose3& map_T_camera,
                                double weight)
{
    EstimatedObject::Ptr new_obj =
      EstimatedObject::Create(graph_,
                              essential_graph_,
                              params_,
                              object_models_[measurement.obj_name],
                              estimated_objects_.size(),
                              n_landmarks_,
                              measurement,
                              map_T_camera, /* G_T_C */
                              I_T_C_,
                              "zed",
                              camera_calibration_,
                              this);

    new_obj->addKeypointMeasurements(measurement, weight);
    n_landmarks_ += new_obj->numKeypoints();
    estimated_objects_.push_back(new_obj);

    ROS_INFO_STREAM(fmt::format(
      "Initializing new object {}, weight {}.", measurement.obj_name, weight));

    object_track_ids_[measurement.track_id] = estimated_objects_.size() - 1;

    return true;
}

bool
SemanticMapper::addMeasurementsToObjects(
  SemanticKeyframe::Ptr kf,
  const aligned_vector<ObjectMeasurement>& measurements,
  const std::vector<size_t>& measurement_index,
  const std::map<size_t, size_t>& known_das,
  const Eigen::MatrixXd& weights,
  const std::vector<size_t>& object_index)
{
    if (measurements.size() == 0)
        return true;

    Pose3 map_T_body = kf->pose();
    Pose3 map_T_camera = map_T_body * I_T_C_;

    /** new objects **/

    for (size_t k = 0; k < measurement_index.size(); ++k) {
        // count the number of observed keypoints
        int n_observed_keypoints = 0;
        for (auto& kp_msmt :
             measurements[measurement_index[k]].keypoint_measurements) {
            if (kp_msmt.observed) {
                n_observed_keypoints++;
            }
        }

        if (weights(k, weights.cols() - 1) >=
              params_.new_landmark_weight_threshold &&
            n_observed_keypoints >=
              params_.min_observed_keypoints_to_initialize) {

            createNewObject(measurements[measurement_index[k]],
                            map_T_camera,
                            weights(k, weights.cols() - 1));
        }
    }
    /** existing objects **/

    // existing objects that were tracked
    for (const auto& known_da : known_das) {
        auto& msmt = measurements[known_da.first];

        ROS_INFO_STREAM(
          fmt::format("Measurement {} [{}]: adding factors from {} to object "
                      "{} [{}] (tracked).",
                      known_da.first,
                      msmt.obj_name,
                      DefaultKeyFormatter(msmt.observed_key),
                      known_da.second,
                      estimated_objects_[known_da.second]->obj_name()));

        estimated_objects_[known_da.second]->addKeypointMeasurements(msmt, 1.0);
    }

    // existing objects that were associated
    for (size_t k = 0; k < measurement_index.size(); ++k) {
        for (int j = 0; j < weights.cols() - 1; ++j) {
            if (weights(k, j) >= params_.constraint_weight_threshold) {
                auto& msmt = measurements[measurement_index[k]];

                ROS_INFO_STREAM(
                  fmt::format("Measurement {} [{}]: adding factors from {} "
                              "to object {} [{}] with weight {}.",
                              measurement_index[k],
                              msmt.obj_name,
                              DefaultKeyFormatter(msmt.observed_key),
                              object_index[j],
                              estimated_objects_[object_index[j]]->obj_name(),
                              weights(k, j)));

                // loop closure check
                if (kf->index() -
                      static_cast<int>(
                        estimated_objects_[object_index[j]]->last_seen()) >
                    loop_closure_threshold_) {
                    kf->loop_closing() = true;
                }

                estimated_objects_[object_index[j]]->addKeypointMeasurements(
                  msmt, weights(k, j));

                // update information about track ids
                object_track_ids_[msmt.track_id] = object_index[j];
            }
        }
    }

    return true;
}

std::vector<bool>
SemanticMapper::predictVisibleObjects(SemanticKeyframe::Ptr kf)
{
    std::vector<bool> visible(estimated_objects_.size(), false);

    Pose3 map_T_body = kf->pose();

    // Pose3 G_T_C = map_T_body * I_T_C_;

    // Do simple range check
    for (size_t i = 0; i < estimated_objects_.size(); ++i) {
        if (estimated_objects_[i]->bad())
            continue;

        // Pose3 map_T_obj =
        // graph_->getNode<SE3Node>(sym::O(estimated_objects_[i]->id()))->pose();
        Pose3 map_T_obj = estimated_objects_[i]->pose();

        Pose3 body_T_obj = map_T_body.inverse() * map_T_obj;

        if (body_T_obj.translation().norm() <= 75) {
            visible[i] = true;
        }
    }

    return visible;
}

void
SemanticMapper::msgCallback(
  const object_pose_interface_msgs::KeypointDetections::ConstPtr& msg)
{
    // ROS_INFO_STREAM("[SemanticMapper] Received keypoint detection
    // message, t = " << msg->header.stamp);
    received_msgs_++;

    if (received_msgs_ > 1 && msg->header.seq != last_msg_seq_ + 1) {
        ROS_ERROR_STREAM(
          "[SemanticMapper] Error: dropped keypoint message. Expected "
          << last_msg_seq_ + 1 << ", got " << msg->header.seq);
    }
    // ROS_INFO_STREAM("Received relpose msg, seq " << msg->header.seq << ",
    // time " << msg->header.stamp);
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        msg_queue_.push_back(*msg);
    }
    // cv_.notify_all();

    last_msg_seq_ = msg->header.seq;
}

std::vector<EstimatedObject::Ptr>
SemanticMapper::estimated_objects()
{
    return estimated_objects_;
}

SemanticKeyframe::Ptr
SemanticMapper::getKeyframeByIndex(int index)
{
    return keyframes_[index];
}

SemanticKeyframe::Ptr
SemanticMapper::getKeyframeByKey(Key key)
{
    return keyframes_[Symbol(key).index()];
}

void
SemanticMapper::addPresenter(boost::shared_ptr<Presenter> presenter)
{
    presenter->setGraph(graph_);
    presenter->setup();
    presenters_.push_back(presenter);
}
