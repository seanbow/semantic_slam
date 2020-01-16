
#include "semantic_slam/SemanticMapper.h"

#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/GeometricFeatureHandler.h"
#include "semantic_slam/LocalParameterizations.h"
#include "semantic_slam/LoopCloser.h"
#include "semantic_slam/MLDataAssociator.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/Presenter.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/SemanticSmoother.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/keypoints/EstimatedKeypoint.h"
#include "semantic_slam/keypoints/EstimatedObject.h"

#include <ros/package.h>

#include <boost/filesystem.hpp>
#include <thread>
#include <unordered_set>

#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/Values.h>

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

    loop_closer_ = util::allocate_aligned<LoopCloser>(this);

    received_msgs_ = 0;
    measurements_processed_ = 0;
    n_landmarks_ = 0;

    running_ = false;

    node_chr_ = 'o';

    gravity_ << 0, 0, -9.81;

    std::string base_path = ros::package::getPath("semantic_slam");
    std::string model_dir = base_path + std::string("/models/objects/");

    loadModelFiles(model_dir);
    loadCalibration();
    loadParameters();

    smoother_ = util::allocate_aligned<SemanticSmoother>(params_, this);
    smoother_->setLoopCloser(loop_closer_);

    operation_mode_ = OperationMode::NORMAL;

    smoother_->setVerbose(verbose_optimization_);
    smoother_->setCovarianceDelay(covariance_delay_);
    smoother_->setMaxOptimizationTime(max_optimization_time_);
    smoother_->setSmoothingLength(smoothing_length_);
    smoother_->setLoopClosureThreshold(loop_closure_threshold_);

    // Only used if our odometry source is inertial
    gravity_node_ = util::allocate_aligned<Vector3dNode>(Symbol('r', 0));
}

void
SemanticMapper::setOdometryHandler(boost::shared_ptr<OdometryHandler> odom)
{
    odometry_handler_ = odom;
    odom->setGraph(smoother_->graph());
    odom->setMapper(this);
    odom->setEssentialGraph(smoother_->essential_graph());
    odom->setup();
}

void
SemanticMapper::setGeometricFeatureHandler(
  boost::shared_ptr<GeometricFeatureHandler> geom)
{
    geom_handler_ = geom;
    geom->setGraph(smoother_->graph());
    geom->setMapper(this);
    geom->setEssentialGraph(smoother_->essential_graph());
    geom->setExtrinsicCalibration(I_T_C_);
    geom->setup();

    if (include_geometric_features_) {
        smoother_->setGeometricFeatureHandler(geom);
    }
}

void
SemanticMapper::start()
{
    running_ = true;

    anchorOrigin();

    std::thread process_messages_thread(
      &SemanticMapper::processMessagesUpdateObjectsThread, this);

    smoother_->start();

    process_messages_thread.join();
    smoother_->stop();
    smoother_->join();

    running_ = false;
}

void
SemanticMapper::processMessagesUpdateObjectsThread()
{
    while (ros::ok() && running_) {

        // bool processed_msg = false;

        if (operation_mode_ == OperationMode::LOOP_CLOSING) {
            if (checkLoopClosingDone())
                processLoopClosure();
        }

        bool did_anything = false;

        if (haveNextKeyframe()) {
            SemanticKeyframe::Ptr next_keyframe = tryFetchNextKeyframe();

            if (next_keyframe) {
                pending_keyframes_.push_back(next_keyframe);
            }

            did_anything = true;
        }

        if (!pending_keyframes_.empty()) {
            int n_keyframes_processed = processPendingKeyframes();
            did_anything = n_keyframes_processed > 0;
        }

        if (did_anything) {
            std::unique_lock<std::mutex> lock(map_mutex_, std::defer_lock);
            std::unique_lock<std::mutex> present_lock(present_mutex_,
                                                      std::defer_lock);
            std::lock(lock, present_lock);

            for (auto& p : presenters_)
                p->present(keyframes_, estimated_objects_);

            std::cout << "Bias 0 = " << keyframes_[0]->bias().transpose()
                      << std::endl;
            // std::cout << "Gravity = " << gravity_.transpose() << std::endl;
            std::cout << "q0 = "
                      << keyframes_[0]->pose().rotation().coeffs().transpose()
                      << std::endl;
        } else {
            ros::Duration(0.001).sleep();
        }
    }

    running_ = false;
}

int
SemanticMapper::processPendingKeyframes()
{
    if (pending_keyframes_.empty() ||
        operation_mode_ == OperationMode::LOOP_CLOSING) {
        // TODO figure out a better way to handle this when LOOP_CLOSING
        return 0;
    }

    int n_keyframes_processed = 0;

    // Use a heuristic to prevent the tracking part from getting too far
    // ahead...
    // TODO think about this more
    // Say we can go until we get half of our smoothing window ahead of the last
    // optimized keyframe
    int last_added_kf_index = pending_keyframes_.front()->index();

    while (
      !pending_keyframes_.empty() &&
      (last_added_kf_index - smoother_->mostRecentOptimizedKeyframeIndex()) <
        smoothing_length_ / 2) {
        auto next_keyframe = pending_keyframes_.front();
        pending_keyframes_.pop_front();

        n_keyframes_processed++;

        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            updateKeyframeObjects(next_keyframe);
        }

        last_added_kf_index = next_keyframe->index();

        if (operation_mode_ == OperationMode::NORMAL &&
            next_keyframe->loop_closing()) {
            ROS_WARN("LOOP CLOSURE!!");

            // because of the asynchronous nature of how measurements are
            // associated and keyframe created (this thread) and how these
            // factors are actually added to the graph (other thread), we can't
            // start the loop closer quite yet as the graph won't contain the
            // loop closing measurements!! need to wait for other thread to
            // incorporate this frame into the graph, so set it to a "pending"
            // status to notify the other thread of this
            operation_mode_ = OperationMode::LOOP_CLOSURE_PENDING;
        }
    }

    return n_keyframes_processed;
}

bool
SemanticMapper::checkLoopClosingDone()
{
    return operation_mode_ == OperationMode::LOOP_CLOSING &&
           !loop_closer_->running();
}

bool
SemanticMapper::processLoopClosure()
{
    if (loop_closer_->running())
        return false;

    // Loop closer is done, update our map with its result
    // If we're currently in the middle of a local optimization, it will likely
    // finish after we perform this update, and try to update the graph with
    // stale values. Set an invalidation flag to prevent that from happening
    smoother_->informLoopClosure();

    std::lock_guard<std::mutex> map_lock(map_mutex_);
    loop_closer_->updateLoopInMapper();

    // Recompute the data association for the post-closure frames
    // TODO figure out how best to account for newly created objects that
    // ideally would not have been created, maybe remove all measurements from
    // the entire set of frames then re-add them
    //
    // covariance information is very wonky here??
    // for (int kf_id = loop_closer_->endOfLoopIndex() + 1;
    //      kf_id < keyframes_.size();
    //      ++kf_id) {

    //     auto kf = getKeyframeByIndex(kf_id);

    //     // only for those frames that have already been processed once,
    //     // otherwise they remain part of the normal order of operations
    //     if (kf->measurements_processed()) {
    //         updateKeyframeObjects(kf);
    //         smoother_->informKeyframeUpdated(kf);
    //     }
    // }

    operation_mode_ = OperationMode::NORMAL;

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

    std::string odometry_source;
    if (!pnh_.getParam("odometry_type", odometry_source)) {
        ROS_ERROR("Unable to load odometry_type parameter");
    }

    if (odometry_source == "external") {
        params_.odometry_source = OdometrySource::EXTERNAL;
    } else if (odometry_source == "inertial") {
        params_.odometry_source = OdometrySource::INERTIAL;
    } else {
        ROS_ERROR_STREAM("Unknown odometry source type: " << odometry_source);
    }

    return true;
}

bool
SemanticMapper::keepFrame(
  const object_pose_interface_msgs::KeypointDetections& msg)
{
    if (keyframes_.empty())
        return true;

    auto last_keyframe = keyframes_.back();

    Pose3 relpose;
    bool got_relpose =
      odometry_handler_->getRelativePoseEstimateTo(msg.header.stamp, relpose);

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
    SemanticKeyframe::Ptr origin_kf = odometry_handler_->originKeyframe();

    keyframes_.emplace_back(origin_kf);
    pending_keyframes_.emplace_back(origin_kf);

    smoother_->setOrigin(origin_kf);
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
                    next_keyframe->covariance() =
                      smoother_->mostRecentKeyframeCovariance();
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

void
SemanticMapper::computeDataAssociationWeights(SemanticKeyframe::Ptr frame)
{
    frame->association_weights().clear();
    frame->association_weights().resize(frame->measurements().size());

    if (frame->measurements().size() > 0) {

        // Create the list of measurements we need to associate.
        // Identify which measurements have been tracked from already known
        // objects
        std::vector<size_t> measurement_index;
        std::map<size_t, size_t> known_das;

        for (size_t i = 0; i < frame->measurements().size(); ++i) {
            frame->association_weights()[i].clear();

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

        Eigen::MatrixXd weights =
          MLDataAssociator(params_).computeConstraintWeights(mahals);

        // Have the weights in a matrix form now, compute vector of weight
        // mappings for each measurement

        for (size_t k = 0; k < measurement_index.size(); ++k) {
            // count the number of observed keypoints
            int n_observed_keypoints = 0;
            for (auto& kp_msmt : frame->measurements()[measurement_index[k]]
                                   .keypoint_measurements) {
                if (kp_msmt.observed) {
                    n_observed_keypoints++;
                }
            }

            if (weights(k, weights.cols() - 1) >=
                  params_.new_landmark_weight_threshold &&
                n_observed_keypoints >=
                  params_.min_observed_keypoints_to_initialize) {
                frame->association_weights()[measurement_index[k]][-1] =
                  weights(k, weights.cols() - 1);
            }
        }

        // existing objects that were tracked
        for (const auto& known_da : known_das) {
            ROS_INFO_STREAM(fmt::format(
              "Measurement {} [{}]: adding factors from {} to object "
              "{} [{}] (tracked).",
              known_da.first,
              frame->measurements()[known_da.first].obj_name,
              DefaultKeyFormatter(
                frame->measurements()[known_da.first].observed_key),
              known_da.second,
              estimated_objects_[known_da.second]->obj_name()));
            frame->association_weights()[known_da.first][known_da.second] = 1.0;
        }

        // existing objects that were associated
        for (size_t k = 0; k < measurement_index.size(); ++k) {
            for (int j = 0; j < weights.cols() - 1; ++j) {
                if (weights(k, j) >= params_.constraint_weight_threshold) {
                    auto& msmt = frame->measurements()[measurement_index[k]];

                    ROS_INFO_STREAM(fmt::format(
                      "Measurement {} [{}]: adding factors from {} "
                      "to object {} [{}] with weight {}.",
                      measurement_index[k],
                      msmt.obj_name,
                      DefaultKeyFormatter(msmt.observed_key),
                      object_index[j],
                      estimated_objects_[object_index[j]]->obj_name(),
                      weights(k, j)));

                    // loop closure check
                    if (frame->index() -
                          static_cast<int>(
                            estimated_objects_[object_index[j]]->last_seen()) >
                        loop_closure_threshold_) {
                        frame->loop_closing() = true;
                    }

                    frame->association_weights()[measurement_index[k]]
                                                [object_index[j]] =
                      weights(k, j);

                    // update information about track ids
                    object_track_ids_[msmt.track_id] = object_index[j];
                }
            }
        }
    }
}

int
SemanticMapper::createNewObject(const ObjectMeasurement& measurement,
                                const Pose3& map_T_camera,
                                double weight)
{
    EstimatedObject::Ptr new_obj =
      EstimatedObject::Create(smoother_->graph(),
                              smoother_->essential_graph(),
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

    return estimated_objects_.size() - 1;
}

bool
SemanticMapper::addMeasurementsToObjects(SemanticKeyframe::Ptr frame)
{
    if (frame->measurements().size() == 0)
        return true;

    Pose3 map_T_body = frame->pose();
    Pose3 map_T_camera = map_T_body * I_T_C_;

    for (size_t msmt_id = 0; msmt_id < frame->measurements().size();
         ++msmt_id) {
        auto& msmt = frame->measurements()[msmt_id];

        int new_obj_id = -1; // set to the id >= 0 if a new object is created

        for (auto& da_pair : frame->association_weights()[msmt_id]) {
            if (da_pair.first == -1) {
                new_obj_id =
                  createNewObject(msmt, map_T_camera, da_pair.second);
            } else {
                estimated_objects_[da_pair.first]->addKeypointMeasurements(
                  msmt, da_pair.second);
            }
        }

        // Change the "new object" signalling id (-1) to the id of the object
        // that was created and this is actually associated with now
        if (new_obj_id >= 0) {
            frame->association_weights()[msmt_id][new_obj_id] =
              frame->association_weights()[msmt_id][-1];

            frame->association_weights()[msmt_id].erase(-1);
        }
    }

    return true;
}

bool
SemanticMapper::removeMeasurementsFromObjects(SemanticKeyframe::Ptr frame)
{
    if (frame->measurements().size() == 0 ||
        frame->association_weights().size() == 0) {
        return false;
    }

    // Similar code to adding measurements except we're removing...
    for (size_t msmt_id = 0; msmt_id < frame->measurements().size();
         ++msmt_id) {
        auto& msmt = frame->measurements()[msmt_id];

        for (auto& da_pair : frame->association_weights()[msmt_id]) {
            ROS_INFO_STREAM("Removing from object " << da_pair.first);
            estimated_objects_[da_pair.first]->removeKeypointMeasurements(msmt);
        }
    }

    return true;
}

bool
SemanticMapper::updateKeyframeObjects(SemanticKeyframe::Ptr frame)
{
    if (!frame)
        return false;

    // TODO fix this bad approximation
    frame->covariance() = smoother_->mostRecentKeyframeCovariance();

    removeMeasurementsFromObjects(frame);

    computeDataAssociationWeights(frame);

    addMeasurementsToObjects(frame);

    // calling the "update" method on each of our objects allows them to
    // remove themselves from the estimation if they're poorly localized
    // and haven't been observed recently. These objects can cause poor
    // data association and errors down the road otherwise
    for (auto& obj : estimated_objects_) {
        if (!obj->bad()) {
            obj->update(frame);
        }
    }

    frame->updateConnections();
    frame->measurements_processed() = true;

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

    if (kf_old->index() == kf->index()) {
        return global_covs;
    }

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

// Eigen::MatrixXd
// SemanticMapper::computePlxExact(Key obj_key, Key x_key)
// {
//     boost::shared_ptr<gtsam::NonlinearFactorGraph> gtsam_graph;
//     boost::shared_ptr<gtsam::Values> gtsam_values;

//     {
//         std::lock_guard<std::mutex> lock(graph_mutex_);

//         gtsam_graph = graph_->getGtsamGraph();
//         gtsam_values = graph_->getGtsamValues();
//     }

//     auto origin_factor =
//       util::allocate_aligned<gtsam::NonlinearEquality<gtsam::Pose3>>(
//         getKeyframeByIndex(0)->key(),
//         gtsam_values->at<gtsam::Pose3>(symbol_shorthand::X(0)));

//     gtsam_graph->push_back(origin_factor);

//     auto obj = getObjectByKey(obj_key);
//     auto kf = getKeyframeByKey(x_key);

//     int Plx_dim = 3 * obj->keypoints().size() + 6;

//     // If the obejct isn't in the graph we can't do this...
//     if (!obj->inGraph()) {
//         return Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);
//     }

//     // Actually we should ignore x_key and get the latest keyframe in the
//     // graph
//     x_key = getLastKeyframeInGraph()->key();

//     gtsam::KeyVector keys;
//     for (size_t i = 0; i < obj->keypoints().size(); ++i) {
//         keys.push_back(obj->keypoints()[i]->key());
//     }
//     keys.push_back(x_key);

//     try {
//         gtsam::Marginals marginals(*gtsam_graph, *gtsam_values);

//         auto joint_cov = marginals.jointMarginalCovariance(keys);

//         return joint_cov.fullMatrix();

//     } catch (gtsam::IndeterminantLinearSystemException& e) {
//         ROS_WARN_STREAM("Covariance computation failed! Error: " <<
//         e.what());

//         return Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);
//     }
// }

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
    received_msgs_++;

    if (received_msgs_ > 1 && msg->header.seq != last_msg_seq_ + 1) {
        ROS_ERROR_STREAM(
          "[SemanticMapper] Error: dropped keypoint message. Expected "
          << last_msg_seq_ + 1 << ", got " << msg->header.seq);
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        msg_queue_.push_back(*msg);
    }

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
    presenter->setGraph(smoother_->graph());
    presenter->setup();
    presenters_.push_back(presenter);
}
