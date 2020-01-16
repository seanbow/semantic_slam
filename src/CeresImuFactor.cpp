#include "semantic_slam/CeresImuFactor.h"

#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/ceres_cost_terms/ceres_inertial.h"
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

CeresImuFactor::CeresImuFactor(boost::shared_ptr<SE3Node> pose0,
                               boost::shared_ptr<Vector3dNode> vel0,
                               boost::shared_ptr<VectorNode<6>> bias0,
                               boost::shared_ptr<SE3Node> pose1,
                               boost::shared_ptr<Vector3dNode> vel1,
                               boost::shared_ptr<VectorNode<6>> bias1,
                               boost::shared_ptr<Vector3dNode> gravity,
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
    nodes_.push_back(gravity);

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
                                bias_node(1)->vector().data(),
                                gravity_node()->vector().data());

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
    // For covariance purposes we will just include a between factor for now...
    auto cost_term = static_cast<InertialCostTerm*>(cf_);

    // The preintegrated rotation is equal to the delta rotation. Position needs
    // an offset based on gravity & rotation estimate
    //
    // p_pre = x0_q_map * (p1 - p0 - v*dt - 0.5*g*dt^2)
    // --> dp = x0_q_map*(p1 - p0) = p_pre + x0_q_map*(v*dt + 0.5*g*dt^2)
    Eigen::Quaterniond x0_q_x1(cost_term->preint_x().head<4>());
    Eigen::Vector3d map_v_x0 = cost_term->preint_x().segment<3>(4);
    Eigen::Vector3d preint_p = cost_term->preint_x().tail<3>();

    double dt = pose_node(1)->time()->toSec() - pose_node(0)->time()->toSec();

    Eigen::Quaterniond x0_q_map = pose_node(0)->pose().rotation().conjugate();

    Eigen::Vector3d x0_p_x1 =
      preint_p +
      x0_q_map * (dt * map_v_x0 + 0.5 * dt * dt * gravity_node()->vector());

    Pose3 between(x0_q_x1, x0_p_x1);

    // For the covariance, we'll assume for now that the "between" covariance is
    // equal to the preintegration covariance (not true but whatever)
    Eigen::MatrixXd P_between(6, 6);
    P_between.block<3, 3>(0, 0) = cost_term->preint_P().block<3, 3>(0, 0);
    P_between.block<3, 3>(3, 3) = cost_term->preint_P().block<3, 3>(6, 6);
    P_between.block<3, 3>(0, 3) = cost_term->preint_P().block<3, 3>(0, 6);
    P_between.block<3, 3>(3, 0) = cost_term->preint_P().block<3, 3>(6, 0);

    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(P_between);

    gtsam_factor_ = util::allocate_aligned<gtsam::BetweenFactor<gtsam::Pose3>>(
      pose_node(0)->key(),
      pose_node(1)->key(),
      gtsam::Pose3(between),
      gtsam_noise);

    graph->push_back(gtsam_factor_);
}