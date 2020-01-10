#include "semantic_slam/ceres_cost_terms/ceres_inertial.h"

#include "semantic_slam/inertial/InertialIntegrator.h"

InertialCostTerm::InertialCostTerm(
  double t0,
  double t1,
  boost::shared_ptr<InertialIntegrator> integrator)
  : t0_(t0)
  , t1_(t1)
  , integrator_(integrator)
{}

ceres::CostFunction*
InertialCostTerm::Create(double t0,
                         double t1,
                         boost::shared_ptr<InertialIntegrator> integrator)
{
    return new ceres::
      AutoDiffCostFunction<InertialCostTerm, 15, 7, 3, 6, 7, 3, 6>(
        new InertialCostTerm(t0, t1, integrator));
}