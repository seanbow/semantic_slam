#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/Presenter.h"

class ObjectMeshPresenter : public Presenter 
{
public:
    void setup();

    void present(const std::vector<boost::shared_ptr<SemanticKeyframe>>& keyframes,
                 const std::vector<boost::shared_ptr<EstimatedObject>>& objects);

    using Presenter::Presenter;

private:
    ros::Publisher vis_pub_;
    
    bool show_object_labels_;
};
