#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Presenter.h"

class PosePresenter : public Presenter
{
  public:
    void setup();

    void present(
      const std::vector<boost::shared_ptr<SemanticKeyframe>>& keyframes,
      const std::vector<boost::shared_ptr<EstimatedObject>>& objects);

    using Presenter::Presenter;

  private:
    ros::Publisher pub_pose_;
};
