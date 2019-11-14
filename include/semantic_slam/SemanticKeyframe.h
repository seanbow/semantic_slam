#pragma once

#include <ros/ros.h>

class SemanticKeyframe {
public:
    SemanticKeyframe(ros::Time time, Key key);

    Key key() const { return key_; }
    ros::Time time() const { return time_; }

    aligned_vector<ObjectMeasurement> measurements;

    // std::vector<ObjectMeasurement>& measurements() { return measurements_; }
    std::vector<EstimatedObject::Ptr>& visible_objects() { return visible_objects_; }

    using Ptr = boost::shared_ptr<SemanticKeyframe>;

private:
    ros::Time time_;
    Key key_;

    // aligned_vector<ObjectMeasurement> measurements_;
    std::vector<EstimatedObject::Ptr> visible_objects_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

SemanticKeyframe::SemanticKeyframe(ros::Time time, Key key)
    : time_(time),
      key_(key)
{
    
}