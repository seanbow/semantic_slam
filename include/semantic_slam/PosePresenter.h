#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Presenter.h"

class PosePresenter : public Presenter 
{
public:
    void setup();

    void present(const std::vector<SemanticKeyframe::Ptr>& keyframes,
                 const std::vector<EstimatedObject::Ptr>& objects);

    using Presenter::Presenter;

private:
    ros::Publisher pub_pose_;
};
