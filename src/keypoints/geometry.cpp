
// #include "semantic_slam/keypoints/gtsam/StructureFactor.h"
// #include "semantic_slam/keypoints/gtsam/StructureProjectionFactor.h"
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/Symbol.h"
#include "semantic_slam/ceres_cost_terms/ceres_pose_prior.h"
// #include "omnigraph/omnigraph.h"

#include <eigen3/Eigen/SVD>

#include <ceres/ceres.h>

#include "semantic_slam/LocalParameterizations.h"
#include "semantic_slam/ceres_cost_terms/ceres_structure_projection.h"

// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/inference/Symbol.h>
// #include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
// #include <gtsam/nonlinear/Marginals.h>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/slam/PriorFactor.h>

namespace sym = symbol_shorthand;

#include <iostream>
using std::cout;
using std::endl;

namespace geometry {
ObjectModelBasis
readModelFile(std::string file_name)
{
    // File contains mu and pc
    // mu is 3 x N, pc is (3*k) x N
    // File format:
    // N k
    // mu
    // pc

    std::cout << "Reading file " << file_name << endl;

    std::ifstream f(file_name);

    if (!f) {
        // error opening file
        throw std::runtime_error("Error opening model file " + file_name);
    }

    size_t N, k;
    f >> N >> k;

    ObjectModelBasis model;
    model.mu = Eigen::MatrixXd(3, N);
    model.pc = Eigen::MatrixXd(3 * k, N);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < N; ++j) {
            f >> model.mu(i, j);
        }
    }

    for (size_t i = 0; i < 3 * k; ++i) {
        for (size_t j = 0; j < N; ++j) {
            f >> model.pc(i, j);
        }
    }

    return model;
}

Eigen::MatrixXd
centralize(const Eigen::MatrixXd& M)
{
    Eigen::VectorXd mean = M.rowwise().mean();

    return M.colwise() - mean;
}

Eigen::MatrixXd
reshapeS_b2v(const Eigen::MatrixXd& S)
{
    int F = S.rows();
    int P = S.cols();

    // S is a set of matrices of size 3xP stacked on top of each other
    // --> there are F/3 of these matrices
    // Turn each of these into a 3*P dimension vector and then
    // stack them side-by side
    // --> result is 3*P by F/3
    int n_rows_result = 3 * P;
    int n_cols_result = F / 3;
    Eigen::MatrixXd result(n_rows_result, n_cols_result);

    for (int block = 0; block < F / 3; block++) {
        // This block corresponds to rows (3*block) through (3*block + 2)
        for (int row = 0; row < 3; ++row) {
            // block size is 3*P...
            result.block(row * P, block, P, 1) =
              S.block(3 * block + row, 0, 1, P).transpose();
        }
    }

    return result;
}

Eigen::MatrixXd
composeShape(const Eigen::MatrixXd& B, const Eigen::VectorXd& C)
{
    size_t N = B.cols();
    size_t k = B.rows() / 3;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(3, N);

    for (size_t i = 0; i < k; ++i) {
        result += C(i) * B.block(3 * i, 0, 3, N);
    }

    return result;
}

// computes the stdev of each row of the input
Eigen::VectorXd
sample_stddev(const Eigen::MatrixXd& data)
{
    // compute the std normalized by N (not N-1)
    Eigen::MatrixXd centered = data.colwise() - data.rowwise().mean();

    return (centered.array().square().rowwise().sum() / data.cols()).sqrt();

    // Eigen::VectorXd result(data.rows());
    // for (int i = 0; i < data.rows(); ++i) {
    //     result(i) = std::sqrt( (centered.row(i) *
    //     centered.row(i).transpose()) / data.cols() );
    // }

    // return result;
}

#include <fstream>
using std::endl;

StructureResult
optimizeStructureFromProjection(const Eigen::MatrixXd& normalized_coords,
                                const geometry::ObjectModelBasis& model,
                                Eigen::VectorXd weights,
                                bool compute_covariance)
{
    size_t m = normalized_coords.cols();
    size_t k = model.pc.rows() / 3;

    Eigen::MatrixXd mu = centralize(model.mu);
    Eigen::MatrixXd pc = centralize(model.pc);

    StructureResult result;

    double lambda = 10;

    //   // Make sure no weights are < 0
    for (int i = 0; i < weights.size(); ++i) {
        if (weights(i) < 0)
            weights(i) = 0;
    }

    // Normalize weights to 1
    weights /= weights.sum();

    ceres::Problem problem;

    ceres::CostFunction* proj_cost = StructureProjectionCostTerm::Create(
      normalized_coords, model, weights, lambda);

    Eigen::VectorXd Z = 10.0 * Eigen::VectorXd::Ones(m);
    Eigen::VectorXd c = Eigen::VectorXd::Zero(k);
    Pose3 pose;

    // initialize in front of the camera...
    pose.translation()(2) = 5;

    std::vector<double*> params;

    ceres::LocalParameterization* local_param = new SE3LocalParameterization;

    problem.AddParameterBlock(pose.data(), 7, local_param);
    problem.SetParameterization(pose.data(), local_param);
    params.push_back(pose.data());

    // depths
    for (size_t i = 0; i < m; ++i) {
        problem.AddParameterBlock(&Z.data()[i], 1);
        params.push_back(&Z.data()[i]);
    }

    if (k > 0) {
        problem.AddParameterBlock(c.data(), k);
        params.push_back(c.data());
    }

    problem.AddResidualBlock(proj_cost, NULL, params);

    // Right now there's nothing constraining that the object is in front of the
    // camera
    // --> A reflected solution with the object behind the camera might have
    // lower error Hack to get around this: first solve for the pose using
    // coordinate descent which DOES enforce that the object is in front of the
    // camera, then use this solution as a prior in a full LM optimization.
    StructureResult cd_result =
      optimizeStructureFromProjectionCoordinateDescent(
        normalized_coords, model, weights);

    Pose3 prior(Eigen::Quaterniond(cd_result.R), cd_result.t);

    Eigen::Matrix<double, 6, 1> prior_noise;
    prior_noise << 10, 10, 10, 1, 1, 1; // ordering is q,p

    ceres::CostFunction* pose_prior =
      PosePriorCostTerm::Create(prior, prior_noise.asDiagonal());
    problem.AddResidualBlock(pose_prior, NULL, pose.data());

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.minimizer_progress_to_stdout = false;
    ceres::Solve(options, &problem, &summary);

    // std::cout << summary.FullReport() << std::endl;

    result.R = pose.rotation().toRotationMatrix();
    result.t = pose.translation();
    if (k > 0) {
        result.C = c;
    }
    result.Z = Z;

    // Compute covariance if requested
    if (compute_covariance) {
        ceres::Covariance::Options cov_opts;
        ceres::Covariance cov(cov_opts);

        // assemble blocks of depth data pointers
        std::vector<std::pair<const double*, const double*>> blocks;

        for (size_t i = 0; i < m; ++i) {
            blocks.push_back(std::make_pair(&Z.data()[i], &Z.data()[i]));
        }

        if (cov.Compute(blocks, &problem)) {

            result.Z_covariance = Eigen::VectorXd(m);

            for (size_t i = 0; i < m; ++i) {
                cov.GetCovarianceBlock(
                  &Z.data()[i], &Z.data()[i], &result.Z_covariance.data()[i]);
            }
        } else {
            // covariance computation failed!!
            // ROS_WARN_STREAM("Projection covariance failed");
        }
    }

    return result;
}

StructureResult
optimizeStructureFromProjectionCoordinateDescent(
  const Eigen::MatrixXd& normalized_coords,
  const geometry::ObjectModelBasis& model,
  const Eigen::VectorXd& weights)
{
    size_t k = model.pc.rows() / 3;

    Eigen::MatrixXd mu = centralize(model.mu);
    Eigen::MatrixXd pc = centralize(model.pc);

    StructureResult result;

    double lambda = 1;
    double tolerance = 1e-3;

    auto D = weights.asDiagonal();

    // rename things to simplify code
    auto& R = result.R;
    auto& t = result.t;
    auto& Z = result.Z;
    auto& W = normalized_coords;
    auto& C = result.C;

    // initialization
    Eigen::MatrixXd S = mu;
    R = Eigen::Matrix3d::Identity();
    t = W.rowwise().mean() * sample_stddev(R.topRows<2>() * S).mean() /
        sample_stddev(W).mean();
    double weight_sum = weights.sum(); // assume > 0

    // cout << "t0 = " << t.transpose() << endl;

    // there are size(pc, 1) / 3 basis elements
    C = Eigen::VectorXd::Zero(pc.rows() / 3, 1);

    double fval = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < 1000; ++iter) {
        Eigen::MatrixXd RS_plus_t = (R * S).colwise() + t;

        // update depth Z
        Z = (W.array() * RS_plus_t.array()).colwise().sum() /
            (W.array().square().colwise().sum());

        // cout << "Z = " << Z.transpose() << endl;

        // Update t,R by aligning S to W*diag(Z)
        Eigen::MatrixXd Sp = W * Z.asDiagonal();

        // Update t
        t = ((Sp - R * S) * D).rowwise().sum() / weight_sum;

        // Update R
        Eigen::MatrixXd St = Sp.colwise() - t;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          St * D * S.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();

        Eigen::Vector3d reflect(1, 1, sgn((U * V.transpose()).determinant()));

        R = U * reflect.asDiagonal() * V.transpose();

        // update C
        if (k > 0) {
            Eigen::MatrixXd y = reshapeS_b2v(R.transpose() * St - mu);
            Eigen::MatrixXd X = reshapeS_b2v(pc);
            Eigen::JacobiSVD<Eigen::MatrixXd> C_svd(
              X, Eigen::ComputeThinU | Eigen::ComputeThinV);

            Eigen::ArrayXd svs = C_svd.singularValues().array();
            svs = svs / (svs.square() + lambda);
            C = C_svd.matrixV() * svs.matrix().asDiagonal() *
                C_svd.matrixU().transpose() * y;

            // update S with C
            S = mu + composeShape(pc, C);
        }

        double last_fval = fval;
        // for a matrix, squaredNorm returns squared frobenius norm
        fval = ((St - R * S) * weights.array().sqrt().matrix().asDiagonal())
                 .squaredNorm() +
               lambda * C.squaredNorm();

        // cout << "Iter: " << iter << ", fval = " << fval << endl;

        if (std::abs(fval - last_fval) /
              (last_fval + std::numeric_limits<double>::epsilon()) <
            tolerance) {
            break;
        }
    }

    // cout << "Optimization result -- \nR = " << endl << result.R  << "\nt = "
    // << result.t.transpose() << endl; cout << "Z = " << Z.transpose() << endl;
    // cout << "C = " << C.transpose() << endl;

    return result;
}

} // namespace geometry