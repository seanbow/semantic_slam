#include "semantic_slam/CeresImuFactor.h"

#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/ceres_cost_terms/ceres_inertial.h"

CeresImuFactor::CeresImuFactor(boost::shared_ptr<SE3Node> pose0,
                               boost::shared_ptr<Vector3dNode> vel0,
                               boost::shared_ptr<VectorNode<6>> bias0,
                               boost::shared_ptr<SE3Node> pose1,
                               boost::shared_ptr<Vector3dNode> vel1,
                               boost::shared_ptr<VectorNode<6>> bias1,
                               boost::shared_ptr<InertialIntegrator> integrator,
                               double t0,
                               double t1,
                               int tag)
  : CeresFactor(FactorType::INERTIAL, tag)
  , integrator_(integrator)
{
    nodes_.push_back(pose0);
    nodes_.push_back(vel0);
    nodes_.push_back(bias0);
    nodes_.push_back(pose1);
    nodes_.push_back(vel1);
    nodes_.push_back(bias1);

    if (t0 < 0.0 || t1 < 0.0) {
        t0_ = pose0->time()->toSec();
        t1_ = pose1->time()->toSec();
    } else {
        t0_ = t0;
        t1_ = t1;
    }

    cf_ = InertialCostTerm::Create(t0_, t1_, integrator);
}

CeresFactor::Ptr
CeresImuFactor::clone() const
{
    return util::allocate_aligned<CeresImuFactor>(nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  integrator_,
                                                  t0_,
                                                  t1_,
                                                  tag_);
}

CeresImuFactor::~CeresImuFactor()
{
    delete cf_;
}

void
CeresImuFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    active_ = true;

    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(cf_,
                                NULL,
                                pose_node(0)->pose().data(),
                                vel_node(0)->vector().data(),
                                bias_node(0)->vector().data(),
                                pose_node(1)->pose().data(),
                                vel_node(1)->vector().data(),
                                bias_node(1)->vector().data());

    residual_ids_[problem.get()] = residual_id;
}

void
CeresImuFactor::createGtsamFactor() const
{
    // TODO
}

void
CeresImuFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    // TODO
}