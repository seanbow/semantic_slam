#pragma once

#include "semantic_slam/Common.h"
#include "semantic_slam/CeresNode.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

// #include "semantic_slam/ceres_quaternion_parameterization.h"
#include "semantic_slam/pose_math.h"

template <typename Vector>
class VectorNode : public CeresNode
{
EIGEN_STATIC_ASSERT_VECTOR_ONLY(Vector);

public:
    VectorNode(Symbol sym, boost::optional<ros::Time> time=boost::none);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    const Vector& vector() const { return vector_; }
    Vector& vector() { return vector_; }

    using Ptr = boost::shared_ptr<VectorNode<Vector>>;

private:
    Vector vector_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename Vector>
using VectorNodePtr = typename VectorNode<Vector>::Ptr;

template <typename Vector>
VectorNode<Vector>::VectorNode(Symbol sym, boost::optional<ros::Time> time)
    : CeresNode(sym, time)
{
}

template <typename Vector>
void VectorNode<Vector>::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->AddParameterBlock(vector_.data(), vector_.size());
}

using Vector2dNode = VectorNode<Eigen::Vector2d>;
using Vector2dNodePtr = VectorNode<Eigen::Vector2d>::Ptr;

using Vector3dNode = VectorNode<Eigen::Vector3d>;
using Vector3dNodePtr = VectorNode<Eigen::Vector3d>::Ptr;
