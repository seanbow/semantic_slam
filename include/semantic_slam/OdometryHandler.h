#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Handler.h"

#include <vector>

class SemanticKeyframe;

class OdometryHandler : public Handler
{
  public:
    OdometryHandler()
      : Handler()
    {}

    virtual void setup() = 0;

    virtual boost::shared_ptr<SemanticKeyframe> originKeyframe() = 0;

    virtual boost::shared_ptr<SemanticKeyframe> createKeyframe(
      ros::Time time) = 0;

    virtual void updateKeyframeAfterOptimization(
      boost::shared_ptr<SemanticKeyframe> keyframe_to_update,
      boost::shared_ptr<SemanticKeyframe> optimized_keyframe) = 0;

    virtual bool getRelativePoseEstimateTo(ros::Time t, Pose3& T12) = 0;

  protected:
    std::vector<boost::shared_ptr<SemanticKeyframe>> keyframes_;
};
