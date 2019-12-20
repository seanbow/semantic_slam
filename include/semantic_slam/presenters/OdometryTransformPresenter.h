#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Presenter.h"

#include <tf2_ros/transform_broadcaster.h>

class OdometryTransformPresenter : public Presenter
{
  public:
    void setup();

    void present(
      const std::vector<boost::shared_ptr<SemanticKeyframe>>& keyframes,
      const std::vector<boost::shared_ptr<EstimatedObject>>& objects);

    void publishFunction();

    using Presenter::Presenter;

  private:
    tf2_ros::TransformBroadcaster broadcaster_;
    Pose3 map_T_odom_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
