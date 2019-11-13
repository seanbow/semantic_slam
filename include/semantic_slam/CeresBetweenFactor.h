#pragma once 

#include <ceres/ceres.h>
#include <eigen3/Eigen/Core>
#include "semantic_slam/CeresFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/pose_math.h"

class CeresBetweenFactor : public CeresFactor
{
public:
    CeresBetweenFactor(SE3NodePtr node1, SE3NodePtr node2, Pose3 between, Eigen::MatrixXd covariance, int tag=0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    using This = CeresBetweenFactor;
    using Ptr = boost::shared_ptr<This>;

private: 
    ceres::CostFunction* cf_;

    SE3NodePtr node1_;
    SE3NodePtr node2_;

};
