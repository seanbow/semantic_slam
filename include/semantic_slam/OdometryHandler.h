#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Handler.h"
// #include "semantic_slam/pose_math.h"

#include <vector>

class SemanticKeyframe;

class OdometryHandler : public Handler
{
  public:
    OdometryHandler();

    virtual void setup() = 0;

    virtual boost::shared_ptr<SemanticKeyframe> originKeyframe(
      ros::Time time) = 0;

    virtual boost::shared_ptr<SemanticKeyframe> createKeyframe(
      ros::Time time) = 0;

    virtual bool getRelativePoseEstimate(ros::Time t1,
                                         ros::Time t2,
                                         Pose3& T12) = 0;

  protected:
    std::vector<boost::shared_ptr<SemanticKeyframe>> keyframes_;
};

OdometryHandler::OdometryHandler()
  : Handler()
{}