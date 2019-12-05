#include "semantic_slam/SimpleObjectTracker.h"

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>

SimpleObjectTracker::SimpleObjectTracker()
    : next_object_id_(0),
        pnh_("~")
{
    image_transport::ImageTransport it(pnh_);
    img_pub_ = it.advertise("tracking_visualization", 1);
    det_pub_ = pnh_.advertise<darknet_ros_msgs::BoundingBoxes>("tracked_objects", 1);

    std::string image_topic, det_topic;

    pnh_.param("image_topic", image_topic, std::string("camera/rgb/image_raw"));
    pnh_.param("detection_topic", det_topic, std::string("detected_objects_in_image"));

    pnh_.param("detection_conf_threshold", det_conf_thresh_, 0.80);
    pnh_.param("f2f_match_threshold", f2f_match_thresh_, 0.0);

    pnh_.param("missed_detection_threshold", missed_detection_thresh_, 0);

    ROS_INFO_STREAM("Object tracker: subscribing to " << image_topic);
    ROS_INFO_STREAM("Object tracker: subscribing to " << det_topic);

    image_sub_ = boost::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(nh_, image_topic, 10);
    det_sub_ = boost::make_shared<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>>(nh_, det_topic, 10);
    sync_ = boost::make_shared<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>>(*image_sub_, *det_sub_, 10);
    sync_->registerCallback(boost::bind(&SimpleObjectTracker::detectionCallback, this, _1, _2));
}


void SimpleObjectTracker::detectionCallback(const sensor_msgs::ImageConstPtr& image,
                                            const darknet_ros_msgs::BoundingBoxes::ConstPtr& msg)
{
    darknet_ros_msgs::BoundingBoxes new_msg;
    new_msg.header = msg->header;
    new_msg.image_header = image->header;

    // ROS_INFO_STREAM("Got detection!!");

    // Mark which objects were observed in this image so we can delete unobserved ones
    std::vector<bool> detected_set(tracked_objects_.size(), 0);

    for (int i = 0; i < msg->bounding_boxes.size(); ++i) {
        auto& det = msg->bounding_boxes[i];

        if (det.probability < det_conf_thresh_) {
            continue;
        }

        double width = det.xmax - det.xmin;
        double height = det.ymax - det.ymin;
        cv::Rect r(det.xmin, det.ymin, width, height);
        cv::Rect2d bbox(r);

        int idx = getTrackingIndex(det.Class, bbox);

        darknet_ros_msgs::BoundingBox det_msg = det;
        det_msg.header = msg->header;

        if (idx == -1 || idx >= detected_set.size()) {
            // not found, initialize a new object
            TrackedObject object;
            object.object_name = det.Class;
            object.id = next_object_id_++;
            object.n_missed_detections = 0;
            object.bounding_box = bbox;

            tracked_objects_.push_back(object);

            det_msg.id = object.id;
        } else {
            tracked_objects_[idx].bounding_box = bbox;
            tracked_objects_[idx].n_missed_detections = 0;
            detected_set[idx] = true;

            det_msg.id = tracked_objects_[idx].id;
        }

        new_msg.bounding_boxes.push_back(det_msg);
    }

    // Iterate downwards so if we erase index i all indices [0,i) remain the same
    for (int i = detected_set.size() - 1; i >= 0; i--) {
        if (!detected_set[i]) {
            tracked_objects_[i].n_missed_detections++;

            if (tracked_objects_[i].n_missed_detections > missed_detection_thresh_) {
                tracked_objects_.erase(tracked_objects_.begin() + i);
            }
        }
    }

    visualizeTracking(image, new_msg);

    det_pub_.publish(new_msg);
}

void SimpleObjectTracker::visualizeTracking(const sensor_msgs::ImageConstPtr& image,
                                            const darknet_ros_msgs::BoundingBoxes& msg)
{
    cv::Mat img = cv_bridge::toCvCopy(image, "bgr8")->image;

    for (auto& obj : tracked_objects_) {
        cv::Scalar color = cv::Scalar(255, 0, 0);

        cv::rectangle(img, obj.bounding_box, color, 2);
        cv::Point loc = obj.bounding_box.tl();
        loc.y = loc.y + 20;

        std::string label = obj.object_name + std::string(": ") + std::to_string(obj.id);
        cv::putText(img, label, loc, 0, 0.80, cv::Scalar(255,255,255), 2);
    }

    auto img_msg = cv_bridge::CvImage(image->header, "bgr8", img).toImageMsg();

    img_pub_.publish(img_msg);
}
  
int SimpleObjectTracker::getTrackingIndex(const std::string& name, const cv::Rect2d& drect)
{
    int best_index = -1;
    double best_match = -1;

    for (int i = 0; i < tracked_objects_.size(); ++i) {
        // Make sure the name matches
        if (tracked_objects_[i].object_name != name) continue;

        // Check the overlap / match score
        double match = calculateMatchRate(tracked_objects_[i].bounding_box, drect);

        if (match > best_match) {
            best_match = match;
            best_index = i;
        }
    }

    if (best_match > f2f_match_thresh_) {
        return best_index;
    } else {
        return -1;
    }
}

double SimpleObjectTracker::calculateMatchRate(const cv::Rect2d& r1, const cv::Rect2d& r2)
{
  cv::Rect2i ir1(r1), ir2(r2);
  /* calculate center of rectangle #1*/
  cv::Point2i c1(ir1.x + (ir1.width >> 1), ir1.y + (ir1.height >> 1));
  /* calculate center of rectangle #2*/
  cv::Point2i c2(ir2.x + (ir2.width >> 1), ir2.y + (ir2.height >> 1));
    
  double a1 = ir1.area(), a2 = ir2.area(), a0 = (ir1 & ir2).area();
  /* calculate the overlap rate*/
  double overlap = a0 / (a1 + a2 - a0);
  /* calculate the deviation between centers #1 and #2*/
  double deviate = sqrt(powf((c1.x - c2.x), 2) + powf((c1.y - c2.y), 2));
  /* calculate the length of diagonal for the rectangle in average size*/
  double len_diag = sqrt(powf(((ir1.width + ir2.width) >> 1), 2) + powf(((ir1.height + ir2.height) >> 1), 2));
    
  // return overlap;

//   double match = overlap * (1 - deviate/len_diag);
//   std::cout << "MATCH " << match << std::endl;
//   return match;

  /* calculate the match rate. The more overlap, the more matching. Contrary, the more deviation, the less matching*/
  // std::cout << "MATCH RATE " << overlap*len_diag/deviate << std::endl;
  return overlap * len_diag / deviate;
}
  