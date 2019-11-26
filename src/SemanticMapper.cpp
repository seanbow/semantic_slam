
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/MLDataAssociator.h"
#include "semantic_slam/GeometricFeatureHandler.h"
#include "semantic_slam/MultiProjectionFactor.h"

#include <ros/package.h>

#include <boost/filesystem.hpp>
#include <thread>
#include <unordered_set>

#include <visualization_msgs/MarkerArray.h>

using namespace std::string_literals;
namespace sym = symbol_shorthand;

SemanticMapper::SemanticMapper()
    : nh_(),
      pnh_("~")
{
    setup();
}

void SemanticMapper::setup()
{
    ROS_INFO("Starting object handler.");
    std::string object_topic;
    pnh_.param("keypoint_detection_topic", object_topic, "/semslam/img_keypoints"s);

    ROS_INFO_STREAM("[SemanticMapper] Subscribing to topic " << object_topic);

    subscriber_ = nh_.subscribe(object_topic, 1000, &SemanticMapper::msgCallback, this);

    graph_ = util::allocate_aligned<FactorGraph>();

    received_msgs_ = 0;
    measurements_processed_ = 0;
    n_landmarks_ = 0;

    last_kf_covariance_ = Eigen::MatrixXd::Zero(6,6);

    running_ = false;

    node_chr_ = 'o';

    std::string base_path = ros::package::getPath("semantic_slam");
    std::string model_dir = base_path + std::string("/models/objects/");

    loadModelFiles(model_dir);
    loadCalibration();
    loadParameters();


    // vis_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("keypoint_objects/object_markers", 10);
}

void SemanticMapper::setOdometryHandler(boost::shared_ptr<OdometryHandler> odom) {
    odometry_handler_ = odom;
    odom->setGraph(graph_);
    odom->setup();
}

void SemanticMapper::setGeometricFeatureHandler(boost::shared_ptr<GeometricFeatureHandler> geom) {
    geom_handler_ = geom;
    geom->setGraph(graph_);
    geom->setExtrinsicCalibration(I_T_C_);
    geom->setup();
}

void SemanticMapper::start()
{
    running_ = true;

    anchorOrigin();

    // while (ros::ok() && running_) {
    //     updateObjects();
    //     // visualizeObjectMeshes();
    //     for (auto& p : presenters_) p->present(keyframes_, estimated_objects_);
    //     addNewOdometryToGraph();
    //     tryAddObjectsToGraph();

    //     bool did_optimize = tryOptimize();

    //     if (did_optimize) {
    //         computeLandmarkCovariances();
    //     }

    //     ros::Duration(0.003).sleep();
    // }

    std::thread process_messages_thread(&SemanticMapper::processMessagesUpdateObjectsThread, this);
    std::thread graph_optimize_thread(&SemanticMapper::addObjectsAndOptimizeGraphThread, this);

    process_messages_thread.join();
    graph_optimize_thread.join();

    running_ = false;
}

void SemanticMapper::processMessagesUpdateObjectsThread()
{
    while (ros::ok() && running_) {

        bool processed_msg = false;

        if (haveNextKeyframe()) {

            tryFetchNextKeyframe();

            if (next_keyframe_) {

                updateNextKeyframeObjects();

                next_keyframe_->updateConnections();
                next_keyframe_->measurements_processed() = true;

                for (auto& p : presenters_) p->present(keyframes_, estimated_objects_);
            }

        } else {
            ros::Duration(0.01).sleep();
        }
    }

    running_ = false;
}

void SemanticMapper::addObjectsAndOptimizeGraphThread()
{
    while (ros::ok() && running_) {
        auto new_frames = addNewOdometryToGraph();


        if (new_frames.size() > 0) {
            processGeometricFeatureTracks(new_frames);
            tryAddObjectsToGraph();

            freezeNonCovisible(new_frames);
            bool did_optimize = tryOptimize();

            if (did_optimize) {
                unfreezeAll();
                if (needToComputeCovariances()) computeCovariances();
            }
        } else {
            ros::Duration(0.01).sleep();
        }
    }

    running_ = false;
}

void SemanticMapper::processGeometricFeatureTracks(const std::vector<SemanticKeyframe::Ptr>& new_keyframes)
{
    using namespace std::chrono;
    auto t1 = high_resolution_clock::now();

    for (auto& kf : new_keyframes) {
        geom_handler_->addKeyframe(kf);
    }

    geom_handler_->processPendingFrames();

    auto t2 = high_resolution_clock::now();

    ROS_INFO_STREAM("Tracking features took " 
        << duration_cast<microseconds>(t2 - t1).count()/1000.0 << " ms." << std::endl);
}

void SemanticMapper::freezeNonCovisible(const std::vector<SemanticKeyframe::Ptr>& target_frames)
{
    // Iterate over the target frames and their covisible frames, collecting the frames
    // and objects that will remain unfrozen in the graph

    unfrozen_kfs_.clear();
    unfrozen_objs_.clear();

    for (const auto& frame : target_frames) {
        unfrozen_kfs_.insert(frame->index());
        for (auto obj : frame->visible_objects()) {
            unfrozen_objs_.insert(obj->id());
        }

        for (const auto& cov_frame : frame->neighbors()) {
            unfrozen_kfs_.insert(cov_frame.first->index());
            for (auto obj : cov_frame.first->visible_objects()) {
                unfrozen_objs_.insert(obj->id());
            }
        }
    }

    // std::cout << "Unfrozen frames: \n";
    // for (int id : unfrozen_kfs_) {
    //     std::cout << id << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Unfrozen objects: \n";
    // for (int id : unfrozen_objs_) {
    //     std::cout << id << " ";
    // }
    // std::cout << std::endl;

    std::lock_guard<std::mutex> lock(graph_mutex_);

    for (const auto& kf : keyframes_) {
        if (!kf->inGraph()) continue;

        if (unfrozen_kfs_.count(kf->index())) {
            graph_->setNodeVariable(kf->graph_node());
        } else {
            graph_->setNodeConstant(kf->graph_node());
        }
    }

    // now objects
    // Why is this broken??
    /*
    for (const auto& obj : estimated_objects_) {
        if (!obj->inGraph()) continue;

        if (unfrozen_objs_.count(obj->id())) {
            obj->setVariableInGraph();
        } else {
            obj->setConstantInGraph();
        }
    }
    */
}

void SemanticMapper::unfreezeAll()
{
    for (const auto& kf : keyframes_) {
        if (!kf->inGraph()) continue;

        // do NOT unfreeze the first (gauge freedom)
        if (kf->index() > 0)
            graph_->setNodeVariable(kf->graph_node());
    }

    for (const auto& obj : estimated_objects_) {
        if (!obj->inGraph()) continue;

        obj->setVariableInGraph();
    }
}

std::vector<SemanticKeyframe::Ptr>
SemanticMapper::addNewOdometryToGraph()
{
    std::vector<SemanticKeyframe::Ptr> new_frames;

    // The first keyframe is special, anchored at its odometry origin
    // if (keyframes_.size() > 0 && !keyframes_[0]->inGraph()) {
    //     keyframes_[0]->addToGraph(graph_);
    //     graph_->setNodeConstant(keyframes_[0]->graph_node());
    // }

    for (auto& kf : keyframes_) {
        if (!kf->inGraph() && kf->measurements_processed()) {
            kf->addToGraph(graph_);
            new_frames.push_back(kf);
        }
    }

    return new_frames;
}

void SemanticMapper::tryAddObjectsToGraph()
{
    if (!params_.include_objects_in_graph) return;

    for (auto& obj : estimated_objects_) {
        if (obj->readyToAddToGraph()) {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            obj->addToGraph();
        }
    }
}

bool SemanticMapper::needToComputeCovariances()
{
    // if (keyframes_.back()->time() - Plxs_time_ > ros::Duration(2)) {
    //     return true;
    // }

    // if (Plxs_.size() != estimated_objects_.size()) {
    //     return true;
    // }

    // return false;

    return true;
}

void SemanticMapper::computeCovariances()
{
    std::lock_guard<std::mutex> lock(graph_mutex_);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Collect all landmark parameter blocks...
    // TODO limit this somehow
    std::vector<CeresNodePtr> landmark_nodes;
    for (const auto& obj : estimated_objects_) {
        if (obj->inGraph()) { // && unfrozen_objs_.count(obj->id())) {
            for (const auto& kp : obj->getKeypoints()) {
                landmark_nodes.push_back(kp->graph_node());
            }
        }
    }

    // Add the most recent processed keyframe
    bool added_keyframe = false;
    int kf_index = 0;
    for (int i = keyframes_.size() - 1; i >= 0; --i) {
        if (keyframes_[i]->inGraph()) { // && unfrozen_kfs_.count(keyframes_[i]->index())) {
            added_keyframe = true;
            kf_index = i;
            landmark_nodes.push_back(keyframes_[kf_index]->graph_node());
            break;
        }
    }

    bool cov_succeeded = graph_->computeMarginalCovariance(landmark_nodes);

    if (!cov_succeeded) {
        ROS_WARN("Covariance computation failed!");
        return;
    }

    std::lock_guard<std::mutex> map_lock(map_mutex_);

    for (const auto& obj : estimated_objects_) {
        if (obj->inGraph()) { // && unfrozen_objs_.count(obj->id())) {
            for (const auto& kp : obj->getKeypoints()) {
                kp->covariance() = graph_->getMarginalCovariance(kp->graph_node());
            }
        }
    }

    if (added_keyframe) {
        keyframes_[kf_index]->covariance() = graph_->getMarginalCovariance(keyframes_[kf_index]->graph_node());
        last_kf_covariance_ = keyframes_[kf_index]->covariance();
        last_kf_covariance_time_ = keyframes_[kf_index]->time();

        for (const auto& obj : estimated_objects_) {
            if (obj->inGraph()) { // && unfrozen_objs_.count(obj->id())) {
                for (const auto& kp : obj->getKeypoints()) {
                    Plxs_[kp->id()] = graph_->getMarginalCovariance(keyframes_[kf_index]->graph_node(),
                                                                    kp->graph_node());
                }
            }
        }

        Plxs_time_ = keyframes_[kf_index]->time();
        Plxs_index_ = kf_index;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    ROS_INFO_STREAM(fmt::format("Computed covariances in {} ms.", duration.count()/1000.0));
}

void SemanticMapper::prepareGraphNodes() 
{
    std::unique_lock<std::mutex> map_lock(map_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::lock(map_lock, graph_lock);

    for (auto& kf : keyframes_) {
        kf->graph_node()->pose() = kf->pose();
    }

    for (auto& obj : estimated_objects_) {
        obj->prepareGraphNode();
    }
}

void SemanticMapper::commitGraphSolution()
{
    std::unique_lock<std::mutex> map_lock(map_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> graph_lock(graph_mutex_, std::defer_lock);
    std::lock(map_lock, graph_lock);

    for (auto& kf : keyframes_) {
        kf->pose() = kf->graph_node()->pose();
    }

    for (auto& obj : estimated_objects_) {
        obj->commitGraphSolution();
    }
}

bool SemanticMapper::tryOptimize() {
    if (!graph_->modified()) return false;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    bool solve_succeeded;
    prepareGraphNodes();
    
    {  
        std::lock_guard<std::mutex> lock(graph_mutex_);
        solve_succeeded = graph_->solve(false);
    }

    if (solve_succeeded) commitGraphSolution();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    if (solve_succeeded) {
        ROS_INFO_STREAM(fmt::format("Solved {} nodes and {} edges in {:.2f} ms.",
                                    graph_->num_nodes(), graph_->num_factors(), duration.count()/1000.0));
        // for (auto& p : presenters_) p->present();
        return true;
    } else {
        ROS_INFO_STREAM("Graph solve failed");
        return false;
    }
}

void SemanticMapper::loadModelFiles(std::string path) {
    boost::filesystem::path model_path(path);
    for (auto i = boost::filesystem::directory_iterator(model_path); i != boost::filesystem::directory_iterator(); i++) {
        if (!boost::filesystem::is_directory(i->path())) {
            ROS_INFO("Loading %s", i->path().filename().string().c_str());
            std::string file_name = i->path().filename().string();
            std::string class_name = file_name.substr(0, file_name.find('.'));

            object_models_[class_name] = geometry::readModelFile(i->path().string());
        }
        else {
            continue;
        }
    }
    ROS_INFO("Loaded %d models", (int)(object_models_.size()));
}

bool SemanticMapper::loadCalibration() {
    // Read from the ros parameter server...

    double fx, fy, s, u0, v0, k1, k2, p1, p2;
    s = 0.0; //skew
    if (!pnh_.getParam("cam_cx", u0) || 
        !pnh_.getParam("cam_cy", v0) || 
        !pnh_.getParam("cam_d0", k1) ||
        !pnh_.getParam("cam_d1", k2) ||
        !pnh_.getParam("cam_d2", p1) ||
        !pnh_.getParam("cam_d3", p2) || 
        !pnh_.getParam("cam_fx", fx) ||
        !pnh_.getParam("cam_fy", fy)) {
        ROS_ERROR("Error: unable to read camera intrinsic parameters.");
        return false;
    }

    camera_calibration_ = boost::make_shared<CameraCalibration>(fx, fy, s, u0, v0, k1, k2, p1, p2);

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

    I_T_C_ = Pose3(q,p);

    return true;
}

bool SemanticMapper::loadParameters()
{
    if (!pnh_.getParam("keypoint_activation_threshold", params_.keypoint_activation_threshold) ||

        !pnh_.getParam("min_object_n_keypoints", params_.min_object_n_keypoints) ||
        !pnh_.getParam("min_landmark_observations", params_.min_landmark_observations) ||

        !pnh_.getParam("structure_regularization_factor", params_.structure_regularization_factor) ||
        !pnh_.getParam("robust_estimator_parameter", params_.robust_estimator_parameter) ||

        !pnh_.getParam("structure_error_coefficient", params_.structure_error_coefficient) ||
        !pnh_.getParam("include_objects_in_graph", params_.include_objects_in_graph) ||

        !pnh_.getParam("new_landmark_weight_threshold", params_.new_landmark_weight_threshold) ||
        !pnh_.getParam("mahal_thresh_assign", params_.mahal_thresh_assign) ||
        !pnh_.getParam("mahal_thresh_init", params_.mahal_thresh_init) ||
        !pnh_.getParam("keypoint_initialization_depth_sigma", params_.keypoint_initialization_depth_sigma) ||
        !pnh_.getParam("constraint_weight_threshold", params_.constraint_weight_threshold)) {

        ROS_ERROR("Unable to load object handler parameters");
        return false;
    }

    return true;
}

bool SemanticMapper::keepFrame(ros::Time time)
{
    if (keyframes_.empty()) return true;

    auto last_keyframe = keyframes_.back();

    Pose3 relpose;
    bool got_relpose = odometry_handler_->getRelativePoseEstimate(last_keyframe->time(), time, relpose);

    if (!got_relpose) {
        ROS_WARN_STREAM("Too few odometry messages received to get keyframe relative pose");
        return true;
    }

    double translation_threshold = 0.1;
    double rotation_threshold = 10; // degrees

    if (relpose.translation().norm() > translation_threshold ||
        2*std::acos(relpose.rotation().w()) > rotation_threshold) {
        return true;
    } else {
        return false;
    }
}

void SemanticMapper::anchorOrigin()
{
    SemanticKeyframe::Ptr origin_kf = odometry_handler_->originKeyframe(ros::Time(0));

    origin_kf->addToGraph(graph_);
    // graph_->addNode(origin_kf->graph_node());
    graph_->setNodeConstant(origin_kf->graph_node());

    keyframes_.push_back(origin_kf);
}

bool SemanticMapper::haveNextKeyframe()
{
    // Check if we have a candidate keyframe waiting in our message queue
    // This will modify the message queue in that it discards non-keyframes,
    // but it will not remove the keyframe from the queue

    std::lock_guard<std::mutex> lock(queue_mutex_);

    while (!msg_queue_.empty()) {
        const auto& msg = msg_queue_.front();

        if (keepFrame(msg.header.stamp)) {
            return true;
        } else {
            msg_queue_.pop_front();
        }
    }
}

bool SemanticMapper::tryFetchNextKeyframe()
{
    object_pose_interface_msgs::KeypointDetections msg;
    next_keyframe_ = nullptr;

    bool got_msg = false;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        while (!msg_queue_.empty() && !got_msg) {
            msg = msg_queue_.front();

            if (keepFrame(msg.header.stamp)) {
                next_keyframe_ = odometry_handler_->createKeyframe(msg.header.stamp);
                if (next_keyframe_) {
                    next_keyframe_->image_time = msg.header.stamp;
                    got_msg = true;
                } else {
                    // Want to keep this keyframe but odometry handler can't make
                    // it for us yet -- try again later
                    return false;
                }
            }

            msg_queue_.pop_front();
        }
    }

    if (!got_msg) return false;

    next_keyframe_->measurements = processObjectDetectionMessage(msg, next_keyframe_->key());

    return true;
}

bool SemanticMapper::updateNextKeyframeObjects()
{
    if (!next_keyframe_) return false;

    std::lock_guard<std::mutex> lock(map_mutex_);

    keyframes_.push_back(next_keyframe_);

    // TODO fix this bad approximation 
    next_keyframe_->covariance() = last_kf_covariance_;

    if (next_keyframe_->measurements.size() > 0) {

        // Create the list of measurements we need to associate.
        // Identify which measurements have been tracked from already known objects
        std::vector<size_t> measurement_index;
        std::map<size_t, size_t> known_das;

        for (size_t i = 0; i < next_keyframe_->measurements.size(); ++i) 
        {
            auto known_id_it = object_track_ids_.find(next_keyframe_->measurements[i].track_id);
            if (known_id_it == object_track_ids_.end()) {
                // track not known, perform actual data association
                measurement_index.push_back(i);
            } else {
                // track ID known, take data association as given
                known_das[i] = known_id_it->second;
            }
        }

        // Create the list of objects we need to associate.
        // count visible landmarks & create mapping from list of visible to list of all
        std::vector<bool> visible = predictVisibleObjects(next_keyframe_);
        size_t n_visible = 0;
        std::vector<size_t> object_index;
        for (size_t j = 0; j < estimated_objects_.size(); ++j)
        {
            if (visible[j])
            {
                n_visible++;
                estimated_objects_[j]->setIsVisible(next_keyframe_);
                object_index.push_back(j);
            }
        }

        Eigen::MatrixXd mahals = Eigen::MatrixXd::Zero(measurement_index.size(), n_visible);
        for (size_t i = 0; i < measurement_index.size(); ++i)
        {
            // ROS_WARN_STREAM("** Measurement " << i << "**");
            for (size_t j = 0; j < n_visible; ++j)
            {
                // mahals(i,j) = computeMahalanobisDistance(keyframe->measurements[measurement_index[i]],
                                                        //  estimated_objects_[object_index[j]]);
                mahals(i, j) = estimated_objects_[object_index[j]]->computeMahalanobisDistance(
                                                                        next_keyframe_->measurements[measurement_index[i]]);
            }
        }

        // std::cout << "Mahals:\n" << mahals << std::endl;
    
        Eigen::MatrixXd weights_matrix = MLDataAssociator(params_).computeConstraintWeights(mahals);

        updateObjects(next_keyframe_, 
                        next_keyframe_->measurements, 
                        measurement_index, 
                        known_das, 
                        weights_matrix, 
                        object_index);
    }

    return true;
}

aligned_vector<ObjectMeasurement>
SemanticMapper::processObjectDetectionMessage(const object_pose_interface_msgs::KeypointDetections& msg,
                                              Key keyframe_key)
{
    aligned_vector<ObjectMeasurement> object_measurements;

    for (auto detection : msg.detections) {
        auto model_it = object_models_.find(detection.obj_name);
        if (model_it == object_models_.end()) {
            ROS_WARN_STREAM("No object model for class " << detection.obj_name);
        }

        geometry::ObjectModelBasis model = model_it->second;


        // Build vectors etc to optimize structure
        Eigen::MatrixXd normalized_coords(3, detection.x.size());
        Eigen::VectorXd weights(detection.x.size());

        for (size_t i = 0; i < detection.x.size(); ++i)
        {
            Eigen::Vector2d img_coords(detection.x[i], detection.y[i]);
            normalized_coords.block<2,1>(0,i) = camera_calibration_->calibrate(img_coords);
            normalized_coords(2, i) = 1.0;

            weights(i) = detection.probabilities[i];
        }

        geometry::StructureResult result;

        try
        {
            result = geometry::optimizeStructureFromProjection(normalized_coords,
                                                                model,
                                                                weights,
                                                                true /* compute depth covariance */);

            // ROS_WARN_STREAM("Optimized structure.");
            // std::cout << "t = " << result.t.transpose() << std::endl;
        }
        catch (std::exception &e)
        {
            ROS_WARN_STREAM("Structure optimization failed:\n"
                            << e.what());
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
            kp_msmt.normalized_measurement = normalized_coords.block<2,1>(0, i);

            kp_msmt.score = detection.probabilities[i];
            kp_msmt.depth = result.Z(i);

            // covariance estimation in the structure optimization may have failed
            if (result.Z_covariance.size() > 0) {
                kp_msmt.depth_sigma = std::sqrt(result.Z_covariance(i));
            } else {
                // TODO what here?
                kp_msmt.depth_sigma = 5;
            }

            kp_msmt.kp_class_id = i;

            // TODO what here?
            kp_msmt.pixel_sigma = 0.05 * bbox_dim;

            // std::cout << "pixel sigma = " << kp_msmt.pixel_sigma << std::endl;

            if (kp_msmt.score > params_.keypoint_activation_threshold)
            {
                kp_msmt.observed = true;
                n_keypoints_observed++;
            } else {
                kp_msmt.observed = false;
            }

            obj_msmt.keypoint_measurements.push_back(kp_msmt);
        }

        if (n_keypoints_observed > 0) {
            object_measurements.push_back(obj_msmt);
        }
    }

    return object_measurements;
}

Eigen::MatrixXd
SemanticMapper::getPlx(Key key1, Key key2)
{
    // TODO fix this

    auto kf = getKeyframeByKey(key2);

    auto kf_old = getKeyframeByIndex(Plxs_index_);

    const Eigen::MatrixXd& Px1 = kf_old->odometry_covariance();
    const Eigen::MatrixXd& Px2 = kf->odometry_covariance();

    // We're doing a weird trick here... TODO check it
    // Assume that the odometry covariance can be approximated by a linear update,
    // i.e. P2 = H*P1*H' for some H.
    // We don't have access to this H, but we have P1 and P2. Write them as L1*L1' and 
    // L2*L2', so L2*L2' = H*L1*L1'*H'. So we see L2 = H*L1 and can solve for it
    Eigen::LLT<Eigen::MatrixXd> llt1(Px1);
    Eigen::LLT<Eigen::MatrixXd> llt2(Px2);

    Eigen::MatrixXd chol1 = llt1.matrixL();
    Eigen::MatrixXd H12 = Eigen::MatrixXd::Identity(9,9);
    H12.block<6,6>(3,3) = llt2.matrixL() * chol1.inverse();

    auto it = Plxs_.find(Symbol(key1).index());
    if (it != Plxs_.end()) {
        const Eigen::MatrixXd& Plx = it->second;
        return Plx * H12.transpose();
    } else {
        return Eigen::MatrixXd::Zero(9,9);
    }

    // EstimatedKeypoint::Ptr kp = nullptr;

    // // Find the landmark...
    // for (auto& obj : estimated_objects_) {
    //     int id = obj->findKeypointByKey(key2);

    //     if (id > 0) {
    //         kp = obj->keypoints()[id];
    //     }
    // }

    // Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(9,9);

    // if (kp) {
    //     cov.block<3,3>(0,0) = kp->covariance();
    // } 

    // return cov;
}

bool SemanticMapper::updateObjects(SemanticKeyframe::Ptr kf,
                                  const aligned_vector<ObjectMeasurement>& measurements,
                                  const std::vector<size_t>& measurement_index,
                                  const std::map<size_t, size_t>& known_das,
                                  const Eigen::MatrixXd& weights,
                                  const std::vector<size_t>& object_index)
{
    if (measurements.size() == 0) return true;

    Pose3 map_T_body = kf->pose();
    Pose3 map_T_camera = map_T_body * I_T_C_;

    /** new objects **/

    for (size_t k = 0; k < measurement_index.size(); ++k)
    {
        if (weights(k, weights.cols() - 1) >= params_.new_landmark_weight_threshold)
        {
            auto& msmt = measurements[measurement_index[k]];

            EstimatedObject::Ptr new_obj =
                EstimatedObject::create(graph_, params_, object_models_[msmt.obj_name], estimated_objects_.size(),
                                        n_landmarks_, msmt, 
                                        map_T_camera, /* G_T_C */
                                        I_T_C_,       
                                        "zed", camera_calibration_, this);

            new_obj->addKeypointMeasurements(msmt, weights(k, weights.cols() - 1));
            n_landmarks_ += new_obj->numKeypoints();

            estimated_objects_.push_back(new_obj);

            ROS_INFO_STREAM(fmt::format("Measurement {} [{}]: initializing new object {}, weight {}.",
                                        k,
                                        estimated_objects_.size() - 1,
                                        msmt.obj_name,
                                        weights(k, weights.cols() - 1)));

            // ROS_INFO_STREAM("Measurement " << k << ": initializing new object " << estimated_objects_.size() - 1 << ", class "
            //                                 << msmt.obj_name << ", weight " << weights(k, weights.cols() - 1));

            object_track_ids_[msmt.track_id] = estimated_objects_.size() - 1;
        }
    }
    /** existing objects **/

    // existing objects that were tracked
    for (const auto& known_da : known_das) {
        auto& msmt = measurements[known_da.first];

        ROS_INFO_STREAM(fmt::format("Measurement {} [{}]: adding factors from {} to object {} [{}] (tracked).",
                                    known_da.first,
                                    msmt.obj_name,
                                    DefaultKeyFormatter(msmt.observed_key),
                                    known_da.second,
                                    estimated_objects_[known_da.second]->obj_name()));

        // ROS_INFO_STREAM("Measurement " << known_da.first << ": adding factors from "
        //                             << DefaultKeyFormatter(msmt.observed_key) << " to object "
        //                             << known_da.second << " with weight 1 [tracked]");

        estimated_objects_[known_da.second]->addKeypointMeasurements(msmt, 1.0);

    }

    // existing objects that were associated
    for (size_t k = 0; k < measurement_index.size(); ++k)
    {
        for (int j = 0; j < weights.cols() - 1; ++j)
        {
            if (weights(k, j) >= params_.constraint_weight_threshold)
            {
                auto& msmt = measurements[measurement_index[k]];

                ROS_INFO_STREAM(fmt::format("Measurement {} [{}]: adding factors from {} "
                                                "to object {} [{}] with weight {}.",
                                            measurement_index[k],
                                            msmt.obj_name,
                                            DefaultKeyFormatter(msmt.observed_key),
                                            object_index[j],
                                            estimated_objects_[object_index[j]]->obj_name(),
                                            weights(k, j)));

                estimated_objects_[object_index[j]]->addKeypointMeasurements(msmt, weights(k, j));

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
    for (size_t i = 0; i < estimated_objects_.size(); ++i)
    {
        if (estimated_objects_[i]->bad())
            continue;

        // Pose3 map_T_obj = graph_->getNode<SE3Node>(sym::O(estimated_objects_[i]->id()))->pose();
        Pose3 map_T_obj = estimated_objects_[i]->pose();

        Pose3 body_T_obj = map_T_body.inverse() * map_T_obj;

        if (body_T_obj.translation().norm() <= 75)
        {
            visible[i] = true;
        }
    }

    return visible; 
}

void SemanticMapper::msgCallback(const object_pose_interface_msgs::KeypointDetections::ConstPtr& msg)
{
    // ROS_INFO_STREAM("[SemanticMapper] Received keypoint detection message, t = " << msg->header.stamp);
	received_msgs_++;

	if (received_msgs_ > 1 && msg->header.seq != last_msg_seq_ + 1) {
        ROS_ERROR_STREAM("[SemanticMapper] Error: dropped keypoint message. Expected " << last_msg_seq_ + 1 << ", got " << msg->header.seq);
    }
    // ROS_INFO_STREAM("Received relpose msg, seq " << msg->header.seq << ", time " << msg->header.stamp);
    {
    	std::lock_guard<std::mutex> lock(queue_mutex_);
    	msg_queue_.push_back(*msg);
    }
    // cv_.notify_all();

    last_msg_seq_ = msg->header.seq;
}

SemanticKeyframe::Ptr SemanticMapper::getKeyframeByIndex(int index)
{
    return keyframes_[index];
}

SemanticKeyframe::Ptr SemanticMapper::getKeyframeByKey(Key key)
{
    return keyframes_[Symbol(key).index()];
}

// double
// SemanticMapper::computeMahalanobisDistance(const ObjectMeasurement& msmt, 
//                                            const EstimatedObject::Ptr& obj)
// {
//   // if (msmt.class_id != classid_) return std::numeric_limits<double>::max();
//   // ROS_WARN_STREAM("Computing mahal for object " << id());
//   if (msmt.obj_name != obj->obj_name()) {
//     // ROS_WARN_STREAM("Object id " << id() << " name " << obj_name_
//     //                              << " != measurement class " << msmt.obj_name);
//     return std::numeric_limits<double>::max();
//   }

//   std::vector<double> matched_distances;

//   for (size_t i = 0; i < msmt.keypoint_measurements.size(); ++i) {
//     if (!msmt.keypoint_measurements[i].observed)
//       continue;

//     const auto& kp_msmt = msmt.keypoint_measurements[i];

//     int kp_match_index = obj->findKeypointByClass(msmt.keypoint_measurements[i].kp_class_id);

//     if (kp_match_index >= 0) {
//       if (!obj->keypoints()[kp_match_index]->bad()) {
//         double d = computeMahalanobisDistance(kp_msmt, obj->keypoints()[kp_match_index]);
//         matched_distances.push_back(d);
//       }
//     }
//   }

//   if (matched_distances.size() == 0)
//     return std::numeric_limits<double>::max();

//   double distance = 0.0;
//   for (auto& x : matched_distances)
//     distance += x;

//   double factor = mahalanobisMultiplicativeFactor(2 * matched_distances.size());

//   // ROS_INFO_STREAM(" Mahal distance " << distance << " * factor " << factor <<
//   // " = " << distance * factor);

//   return distance * factor;
// }

// double SemanticMapper::computeMahalanobisDistance(const KeypointMeasurement& msmt,
//                                                   const EstimatedKeypoint::Ptr& kp)
// {
//   if (kp->bad())
//   {
//     ROS_WARN_STREAM("Mahalanobis distance called for BAD landmark " << id());
//     return std::numeric_limits<double>::max();
//   }

//     Pose3 G_T_I = keyframes_[Symbol(msmt.measured_key).index()]->pose();

//   Camera camera(G_T_I.compose(I_T_C_));
//   Eigen::Vector2d zhat;

//   try
//   {
//     zhat = camera.project(kp->position());
//   }
//   catch (CheiralityException& e)
//   {
//     return std::numeric_limits<double>::max();
//   }

//   // ROS_WARN_STREAM("Computing mahal for object " << parent_->id() << "; kp " << id());

//   // check if visible
//   // if (fabs(zhat(0)) > 1.25 || fabs(zhat(1)) > 1.25) {
//   //     // clearly not in frame
//   //     return std::numeric_limits<double>::max();
//   // }

//   Eigen::Vector2d residual = msmt.normalized_measurement - zhat;

//   Eigen::Matrix<double, 2, 9> H = computeProjectionJacobian(G_T_I.rotation().toRotationMatrix(), 
//                                                             G_T_I.translation(),
//                                                             I_T_C_.rotation().toRotationMatrix(), 
//                                                             kp->position());

//   Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
//   double px_sigma = msmt.pixel_sigma;
//   R(0, 0) = px_sigma * px_sigma / (camera_calibration_->fx() * camera_calibration_->fx());
//   R(1, 1) = px_sigma * px_sigma / (camera_calibration_->fy() * camera_calibration_->fy());

//   // std::cout << "R = \n" << R << std::endl;

//   // Eigen::Matrix2d S = H * Plx * H.transpose() + R;
//   // double mahal = residual.transpose() * S.inverse() * residual;

//   Eigen::MatrixXd Plx = Eigen::MatrixXd::Zero(9,9);

//   if (!kp->inGraph()) {
//     Eigen::MatrixXd Plx = kp->parent_object()->getPlx(sym::L(id()), Symbol(msmt.measured_key));
//   } else {
//     // TODO TODO ugh
//     Plx.block<3,3>(0,0) = global_covariance_;
//   }


//   // std::cout << "Plx for landmark " << id() << ": " << std::endl;
//   // std::cout << Plx << std::endl;

//   double mahal = residual.transpose() * (H * Plx * H.transpose() + R).lu().solve(residual);

//   return mahal;
// }

void SemanticMapper::visualizeObjectMeshes() const
{
    visualization_msgs::MarkerArray object_markers;
    visualization_msgs::Marker object_marker;
    object_marker.type = visualization_msgs::Marker::MESH_RESOURCE;

    object_marker.header.frame_id = "map";
    object_marker.header.stamp = ros::Time::now();

    double model_scale = 1;

    object_marker.scale.x = model_scale;
    object_marker.scale.y = model_scale;
    object_marker.scale.z = model_scale;

    object_marker.color.r = 0;
    object_marker.color.g = 0;
    object_marker.color.b = 1;
    object_marker.color.a = 1.0f;

    object_marker.ns = "objects";
    object_marker.action = visualization_msgs::Marker::ADD;

    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "map";
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.ns = "objects";
    delete_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    delete_marker.action = visualization_msgs::Marker::DELETE;

    size_t n_added = 0;

    for (const EstimatedObject::Ptr& obj : estimated_objects_) {
        if (obj->bad()) {
            delete_marker.id = obj->id();
            object_markers.markers.push_back(delete_marker);
            continue;
        }

        // if (!obj->inGraph()) continue;

        if (obj->inGraph()) {
            // blue
            object_marker.color.b = 1.0;
            object_marker.color.r = 0.0;
            object_marker.color.a = 1.0;
        } else {
            // red
            object_marker.color.b = 0.0;
            object_marker.color.r = 1.0;
            object_marker.color.a = 0.5;
        }


        object_marker.action = visualization_msgs::Marker::ADD;
        // object_text.action = visualization_msgs::Marker::ADD;

        double model_scale = 1.0;

        if (obj->obj_name() == "chair") {
            model_scale = 2;
        } else if (obj->obj_name() == "gascan") {
            model_scale = 0.25;
        } else if (obj->obj_name() == "cart") {
            model_scale = 0.12;
        } else if (obj->obj_name() == "tableclosed") {
            model_scale = 10.0;
        } else if (obj->obj_name() == "ladder") {
            model_scale = 0.125;
        } else if (obj->obj_name() == "pelican") {
            model_scale = 0.25;
        } 

        object_marker.scale.x = model_scale;
        object_marker.scale.y = model_scale;
        object_marker.scale.z = model_scale;

        object_marker.pose.position.x = obj->pose().x();
        object_marker.pose.position.y = obj->pose().y();
        object_marker.pose.position.z = obj->pose().z();

        Eigen::Quaterniond q = obj->pose().rotation();

        object_marker.pose.orientation.x = q.x();
        object_marker.pose.orientation.y = q.y();
        object_marker.pose.orientation.z = q.z();
        object_marker.pose.orientation.w = q.w();

        object_marker.mesh_resource = std::string("package://semantic_slam/models/viz_meshes/") + obj->obj_name() + ".dae";
        // object_marker.mesh_resource = std::string("package://semslam/models/viz_meshes/car.dae");

        // object_text.pose.position.x = obj->pose().translation().x();
        // object_text.pose.position.y = obj->pose().translation().y();
        // object_text.pose.position.z = Z_SCALE * obj->pose().translation().z() + 1.5;

        object_marker.id = obj->id();
        object_marker.text = obj->obj_name();

        object_markers.markers.push_back(object_marker);


        std::vector<int64_t> keypoint_ids = obj->getKeypointIndices();
        // addKeypointMarkers(object_markers.markers, model_scale, obj, t);

        ++n_added;
    }

    vis_pub_.publish(object_markers);
}

void SemanticMapper::visualizeObjects() const
{
  // Clear screen first
  visualization_msgs::MarkerArray reset_markers;
  reset_markers.markers.resize(1);
  reset_markers.markers[0].header.frame_id = "map";
  reset_markers.markers[0].header.stamp = ros::Time::now();
  reset_markers.markers[0].ns = "deleteAllMarkers";
  reset_markers.markers[0].action = 3;
  vis_pub_.publish(reset_markers);

  visualization_msgs::MarkerArray markers_message;

  double scale = 2;

  // ** Full object markers **

  visualization_msgs::Marker marker;

  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;

  marker.color.r = 0;
  marker.color.g = 0;
  marker.color.b = 1;
  marker.color.a = .8;

  marker.scale.x = scale / 8.0;
  marker.scale.y = scale / 8.0;
  marker.scale.z = scale / 8.0;

  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();

  marker.ns = "objects";

  visualization_msgs::Marker object_text;

  object_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  object_text.header.frame_id = "map";
  object_text.header.stamp = ros::Time::now();
  object_text.action = visualization_msgs::Marker::ADD;
  object_text.ns = "object_names";

  object_text.scale.z = 1;

  object_text.color.r = 0;
  object_text.color.g = 0;
  object_text.color.b = 0;
  object_text.color.a = 1.0f;

  for (size_t i = 0; i < estimated_objects_.size(); ++i)
  {
    if (estimated_objects_[i]->bad())
      continue;

    if (estimated_objects_[i]->inGraph())
    {
      marker.color.r = 0;
      marker.color.g = 1;
      marker.color.b = 0;
      marker.color.a = 1.0;
    }
    else
    {
      marker.color.r = 1;
      marker.color.g = 0;
      marker.color.b = 0;
      marker.color.a = 0.5f;
    }

    marker.id = i;

    Pose3 pose = estimated_objects_[i]->pose();
    marker.pose.position.x = pose.x();
    marker.pose.position.y = pose.y();
    marker.pose.position.z = pose.z();

    markers_message.markers.push_back(marker);

    object_text.id = i;
    object_text.pose = marker.pose;
    object_text.pose.position.z += 1;
    object_text.text = fmt::format("{}-{}", estimated_objects_[i]->obj_name(), i);

    markers_message.markers.push_back(object_text);
  }

//   // ** Component keypoint markers **
//   scale = 1.0;

//   marker.color.g = 1;
//   marker.color.b = 0;
//   marker.color.a = 1;

//   marker.scale.x = scale / 8.0;
//   marker.scale.y = scale / 8.0;
//   marker.scale.z = scale / 8.0;

//   marker.ns = "keypoints";

//   object_text.ns = "keypoint_ids";
//   object_text.scale.z = 0.5;

//   for (auto &o : estimated_objects_)
//   {
//     // if (o->bad() || !o->inGraph()) continue;
//     const auto &kps = o->getKeypoints();

//     for (size_t i = 0; i < kps.size(); ++i)
//     {
//       if (kps[i]->bad())
//       {
//         marker.action = visualization_msgs::Marker::DELETE;
//       }
//       else
//       {
//         marker.action = visualization_msgs::Marker::ADD;
//       }

//       if (kps[i]->inGraph())
//       {
//         marker.color.g = 1;
//         marker.color.r = 0;
//       }
//       else
//       {
//         marker.color.g = 0;
//         marker.color.r = 1;
//       }

//       marker.id = kps[i]->id();

//       marker.pose.position.x = kps[i]->position().x();
//       marker.pose.position.y = kps[i]->position().y();
//       marker.pose.position.z = kps[i]->position().z();

//       markers_message.markers.push_back(marker);

//       object_text.pose.position = marker.pose.position;
//       object_text.pose.position.z += 0.5;
//       object_text.id = kps[i]->id();
//       // object_text.text = fmt::format("{}-{}", o->id(), kps[i]->classid());
//       object_text.text = fmt::format("{}", kps[i]->classid());
//       markers_message.markers.push_back(object_text);
//     }
//   }

  vis_pub_.publish(markers_message);
}

void SemanticMapper::addPresenter(boost::shared_ptr<Presenter> presenter)
{
    presenter->setGraph(graph_);
    presenter->setup();
    presenters_.push_back(presenter);
}
