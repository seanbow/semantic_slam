#pragma once

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/Common.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "semantic_slam/Pose3.h"

#include <gtsam/base/GenericValue.h>
#include <gtsam/base/VectorSpace.h> // for type traits

template<int Dim>
class VectorNode : public CeresNode
{
  public:
    using VectorType = Eigen::Matrix<double, Dim, 1>;
    using This = VectorNode<Dim>;

    VectorNode(Symbol sym,
               boost::optional<ros::Time> time = boost::none,
               size_t runtime_size = 0);

    void addToProblem(boost::shared_ptr<ceres::Problem> problem);

    size_t dim() const { return vector_.size(); }
    size_t local_dim() const { return vector_.size(); }

    const VectorType& vector() const { return vector_; }
    VectorType& vector() { return vector_; }

    boost::shared_ptr<CeresNode> clone() const;

    boost::shared_ptr<gtsam::Value> getGtsamValue() const;

    using Ptr = boost::shared_ptr<This>;

    static constexpr size_t SizeAtCompileTime = Dim;

  private:
    Eigen::Matrix<double, Dim, 1> vector_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<int Dim>
using VectorNodePtr = typename VectorNode<Dim>::Ptr;

template<int Dim>
boost::shared_ptr<gtsam::Value>
VectorNode<Dim>::getGtsamValue() const
{
    return util::allocate_aligned<gtsam::GenericValue<VectorType>>(vector_);
    // return util::allocate_aligned<gtsam::Value>(vector_);
}

template<int Dim>
boost::shared_ptr<CeresNode>
VectorNode<Dim>::clone() const
{
    auto node = util::allocate_aligned<This>(symbol(), time(), vector_.size());
    node->vector() = vector_;
    return node;
}

template<int Dim>
VectorNode<Dim>::VectorNode(Symbol sym,
                            boost::optional<ros::Time> time,
                            size_t runtime_size)
  : CeresNode(sym, time)
{

    if (Dim == Eigen::Dynamic) {
        // if Dim == Eigen::Dynamic, require that a size is passed in at
        // construction time
        if (runtime_size == 0) {
            throw std::runtime_error("Error: dynamic vector nodes must be "
                                     "constructed with a size parameter");
        }

        vector_ = Eigen::Matrix<double, Dim, 1>(runtime_size);
    }

    vector_.setZero();

    addParameterBlock(vector_.data(), vector_.size());

    // parameter_blocks_.push_back(vector_.data());
    // parameter_block_sizes_.push_back(vector_.size());
    // parameter_block_local_sizes_.push_back(vector_.size());
    // local_parameterizations_.push_back(nullptr);
}

template<int Dim>
void
VectorNode<Dim>::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    problem->AddParameterBlock(vector_.data(), vector_.size());
    active_problems_.push_back(problem.get());
    active_ = true;
}

using Vector2dNode = VectorNode<2>;
using Vector2dNodePtr = VectorNode<2>::Ptr;

using Vector3dNode = VectorNode<3>;
using Vector3dNodePtr = VectorNode<3>::Ptr;

using VectorXdNode = VectorNode<Eigen::Dynamic>;
using VectorXdNodePtr = VectorNode<Eigen::Dynamic>::Ptr;