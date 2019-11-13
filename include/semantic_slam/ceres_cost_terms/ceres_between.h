#pragma once

#define AUTODIFF 0

using CeresBetweenFactorPtr = CeresBetweenFactor::Ptr;

class BetweenCostTerm 
#if !(AUTODIFF)
    : public ceres::SizedCostFunction<6, 4, 3, 4, 3>
#endif
{
private:
    typedef BetweenCostTerm This;

public:
    using shared_ptr = boost::shared_ptr<This>;
    using Ptr = shared_ptr;

    BetweenCostTerm(const Pose3& between, const Eigen::MatrixXd& cov);

#if AUTODIFF
    template <typename T>
    bool operator()(const T* const q1, 
                    const T* const p1, 
                    const T* const q2, 
                    const T* const p2, 
                    T* residual_ptr) const;
#else
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
#endif

    static ceres::CostFunction* Create(const Pose3& between, const Eigen::MatrixXd& cov);

private:
    Eigen::Matrix<double, 6, 6> sqrt_information_;

    Eigen::Quaterniond dq_;
    Eigen::Vector3d dp_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



BetweenCostTerm::BetweenCostTerm(const Pose3& between, const Eigen::MatrixXd& cov)
{
    // Sqrt of information matrix
    Eigen::MatrixXd sqrtC = cov.llt().matrixL();
    sqrt_information_.setIdentity();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(sqrt_information_);

    dq_ = between.rotation();
    dp_ = between.translation();
}

#if AUTODIFF

template <typename T>
bool BetweenCostTerm::operator()(const T* const q1_ptr, 
                                 const T* const p1_ptr, 
                                 const T* const q2_ptr, 
                                 const T* const p2_ptr, 
                                 T* residuals_ptr) const
{
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_a(p1_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_a(q1_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_b(p2_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_b(q2_ptr);

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        dq_.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = T(2.0) * delta_q.vec();
    residuals.template block<3, 1>(3, 0) =
        p_ab_estimated - dp_.template cast<T>();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
}

ceres::CostFunction* BetweenCostTerm::Create(const Pose3& between, const Eigen::MatrixXd& cov)
{
    BetweenCostTerm* term = new BetweenCostTerm(between, cov);
    return new ceres::AutoDiffCostFunction<BetweenCostTerm, 6, 4, 3, 4, 3>(term);
}

#else

bool BetweenCostTerm::Evaluate(double const* const* parameters, double* residuals_ptr, double** jacobians) const
{
    Eigen::Map<const Eigen::Quaterniond> q1(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> p1(parameters[1]);

    Eigen::Map<const Eigen::Quaterniond> q2(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> p2(parameters[3]);

    Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals_ptr);

    Pose3 pose1(q1, p1);
    Pose3 pose2(q2, p2);

    Eigen::MatrixXd Hpose1, Hpose2;

    Pose3 between = pose1.between(pose2, Hpose1, Hpose2);

    Eigen::MatrixXd Herror;

    Eigen::Quaterniond between_q_inv = between.rotation().conjugate();
    Eigen::Quaterniond dq = dq_ * between_q_inv;
    Eigen::MatrixXd H_q_error = math::Dquat_mul_dq2(dq_, between_q_inv) * math::Dquat_inv(between.rotation());

    Eigen::Vector3d dp = between.translation() - dp_;

    residual.head<3>() = 2.0 * dq.vec();
    residual.tail<3>() = dp;

    residual.applyOnTheLeft(sqrt_information_);

    // Jacobians with respect to each parameter...
    if (jacobians != NULL) {

        if (jacobians[0] != NULL) {
            // 6x4, w.r.t. q1
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> dr_dq1(jacobians[0]);

            dr_dq1.block<3,4>(0,0) = 2.0 * (H_q_error * Hpose1.block<4,4>(0,0)).block<3,4>(0,0);
            dr_dq1.block<3,4>(3,0) = Hpose1.block<3,4>(4,0);
            dr_dq1.applyOnTheLeft(sqrt_information_);
        }

        if (jacobians[1] != NULL) {
            // 6x3, w.r.t. p1
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> dr_dp1(jacobians[1]);
            dr_dp1.block<3,3>(0,0) = Hpose1.block<3,3>(0,4);
            dr_dp1.block<3,3>(3,0) = Hpose1.block<3,3>(4,4);
            dr_dp1.applyOnTheLeft(sqrt_information_);
        }

        if (jacobians[2] != NULL) {
            // 6x4, w.r.t. q2
            Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> dr_dq2(jacobians[2]);
            dr_dq2.block<3,4>(0,0) = 2.0 * (H_q_error * Hpose2.block<4,4>(0,0)).block<3,4>(0,0);
            dr_dq2.block<3,4>(3,0) = Hpose2.block<3,4>(4,0);
            dr_dq2.applyOnTheLeft(sqrt_information_);
        }

        if (jacobians[3] != NULL) {
            // 6x3, w.r.t. p2
            Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> dr_dp2(jacobians[3]);
            dr_dp2.block<3,3>(0,0) = Hpose2.block<3,3>(0,4);
            dr_dp2.block<3,3>(3,0) = Hpose2.block<3,3>(4,4);
            dr_dp2.applyOnTheLeft(sqrt_information_);
        }
    }

    return true;
}

ceres::CostFunction* BetweenCostTerm::Create(const Pose3& between, const Eigen::MatrixXd& cov)
{
    return new BetweenCostTerm(between, cov);
}

#endif
