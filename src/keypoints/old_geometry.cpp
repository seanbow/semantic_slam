
#include "semantic_slam/keypoints/geometry.h"
#include "semantic_slam/keypoints/gtsam/RangeFactor.h"
#include "semantic_slam/keypoints/gtsam/StructureFactor.h"
#include "semantic_slam/keypoints/gtsam/StructureProjectionFactor.h"
// #include "omnigraph/omnigraph.h"

#include <eigen3/Eigen/SVD>

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>

namespace sym = gtsam::symbol_shorthand;

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

StructureResult
optimizeStructure(const Eigen::MatrixXd& points,
                  const ObjectModelBasis& model,
                  const Eigen::VectorXd& weights)
{
    size_t m = model.mu.cols();
    size_t k = model.pc.rows() / 3;

    Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t0 = Eigen::Vector3d::Zero();

    gtsam::Pose3 pose0 = gtsam::Pose3(gtsam::Rot3(R0), gtsam::Point3(t0));

    std::vector<gtsam::Key> landmark_keys;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    for (size_t i = 0; i < m; ++i) {
        landmark_keys.push_back(sym::L(i));

        values.insert(sym::L(i), gtsam::Point3(points.col(i)));

        // note: some of these points may be garbage estimates and way far from
        // the true values. the caller is responsible for weighting them low.
        // add priors to all points anyway

        Eigen::Vector3d prior_noise = Eigen::Vector3d::Constant(0.05);

        auto prior_model = gtsam::noiseModel::Diagonal::Sigmas(prior_noise);
        gtsam::PriorFactor<gtsam::Point3> prior(
          sym::L(i), gtsam::Point3(points.col(i)), prior_model);
        graph.push_back(prior);
    }

    values.insert(sym::O(0), pose0);

    gtsam::Vector c0 = Eigen::VectorXd::Zero(k);
    values.insert(sym::C(0), c0);

    semslam::StructureFactor sf(
      sym::O(0), landmark_keys, sym::C(0), model, weights);

    graph.push_back(sf);

    gtsam::LevenbergMarquardtParams lm_params;
    // lm_params.setVerbosityLM("SUMMARY");
    // lm_params.setVerbosityLM("DAMPED");
    lm_params.diagonalDamping = true;

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);

    gtsam::Values result_values = optimizer.optimize();

    StructureResult result;

    result.R = result_values.at<gtsam::Pose3>(sym::O(0)).rotation().matrix();
    result.t = result_values.at<gtsam::Pose3>(sym::O(0)).translation().vector();
    result.C = result_values.at<gtsam::Vector>(sym::C(0));

    // cout << "Optimization result -- \nR = " << endl << result.R  << "\nt = "
    // << result.t.transpose() << endl;

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
optimizeStructureFromProjectionWithDepths(
  const Eigen::MatrixXd& normalized_coords,
  const Eigen::VectorXd& depths,
  const ObjectModelBasis& model,
  Eigen::VectorXd weights,
  const ObjectParams& params,
  bool compute_covariance)
{
    size_t m = normalized_coords.cols();
    size_t k = model.pc.rows() / 3;

    // cout << "Optimizing with coords:\n" << normalized_coords << endl;
    // cout << "depths: " << endl << depths.transpose() << endl;
    // cout << "and weights: " << endl << weights.transpose() << endl;
    // cout << endl << endl;
    // cout << "model mu: " << endl << model.mu << endl;
    // cout << "model pc: \n" << model.pc << endl;

    Eigen::MatrixXd mu = centralize(model.mu);
    Eigen::MatrixXd pc = centralize(model.pc);

    double lambda = params.structure_regularization_factor;

    std::vector<gtsam::Key> landmark_keys;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    // Make sure no weights are < 0
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights(i) < 0)
            weights(i) = 0;
    }

    // Normalize weights...?
    weights /= weights.sum();
    weights *= params.structure_error_coefficient;

    gtsam::Vector c0 = Eigen::VectorXd::Zero(k);
    if (k > 0)
        values.insert(sym::C(0), c0);

    // Take input depths as priors on the depth values...
    // TODO what sigma etc
    for (size_t i = 0; i < depths.size(); ++i) {
        landmark_keys.push_back(sym::Z(i));

        if (depths(i) < 0) {
            values.insert(sym::Z(i), 10.0);
            continue;
        }

        double sigma = params.point_cloud_depth_sigma;
        auto prior_model = gtsam::noiseModel::Isotropic::Sigma(1, sigma);
        gtsam::PriorFactor<double> prior(sym::Z(i), depths(i), prior_model);
        graph.push_back(prior);

        values.insert(sym::Z(i), depths(i));
    }

    semslam::StructureProjectionFactor spf(normalized_coords,
                                           sym::O(0),
                                           landmark_keys,
                                           sym::C(0),
                                           model,
                                           weights,
                                           lambda);
    graph.push_back(spf);

    Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t0 = Eigen::Vector3d::Zero();

    gtsam::Pose3 pose0 = gtsam::Pose3(gtsam::Rot3(R0), gtsam::Point3(t0));

    values.insert(sym::O(0), pose0);

    gtsam::LevenbergMarquardtParams lm_params;
    // lm_params.setVerbosityLM("SUMMARY");
    // lm_params.setVerbosityLM("DAMPED");
    lm_params.diagonalDamping = true;

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);

    gtsam::Values lm_result = optimizer.optimize();

    StructureResult result;

    result.R = lm_result.at<gtsam::Pose3>(sym::O(0)).rotation().matrix();
    result.t = lm_result.at<gtsam::Pose3>(sym::O(0)).translation().vector();
    if (k > 0)
        result.C = lm_result.at<gtsam::Vector>(sym::C(0));

    result.Z = Eigen::VectorXd(m);
    for (size_t i = 0; i < m; ++i) {
        result.Z(i) = lm_result.at<double>(sym::Z(i));
    }

    // cout << "Optimization result -- \nR = " << endl << result.R  << "\nt = "
    // << result.t.transpose() << endl; if (k > 0) cout << " c = " <<
    // result.C.transpose() << endl;

    if (compute_covariance) {
        gtsam::Marginals marginals(graph, lm_result);
        result.Z_covariance = Eigen::VectorXd(m);
        for (size_t i = 0; i < m; ++i) {
            result.Z_covariance(i) =
              marginals.marginalCovariance(sym::Z(i))(0, 0);
        }
    }

    return result;
}

StructureResult
optimizeStructureFromProjection(const Eigen::MatrixXd& normalized_coords,
                                const ObjectModelBasis& model,
                                Eigen::VectorXd weights,
                                bool compute_covariance)
{
    size_t m = normalized_coords.cols();
    size_t k = model.pc.rows() / 3;

    // cout << "Optimizing with coords:\n" << normalized_coords << endl;
    // cout << "depths: " << endl << depths.transpose() << endl;
    // cout << "and weights: " << endl << weights.transpose() << endl;
    // cout << endl << endl;
    // cout << "model mu: " << endl << model.mu << endl;
    // cout << "model pc: \n" << model.pc << endl;

    Eigen::MatrixXd mu = centralize(model.mu);
    Eigen::MatrixXd pc = centralize(model.pc);

    StructureResult result;

    double lambda = 10;

    std::vector<gtsam::Key> landmark_keys;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    for (size_t i = 0; i < m; ++i) {
        landmark_keys.push_back(sym::Z(i)); // depths
        values.insert(sym::Z(i), 10.0);
    }

    // Make sure no weights are < 0
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights(i) < 0)
            weights(i) = 0;
    }

    // Normalize weights to 1
    weights /= weights.sum();

    gtsam::Vector c0 = Eigen::VectorXd::Zero(k);
    if (k > 0)
        values.insert(sym::C(0), c0);

    semslam::StructureProjectionFactor spf(normalized_coords,
                                           sym::O(0),
                                           landmark_keys,
                                           sym::C(0),
                                           model,
                                           weights,
                                           lambda);
    graph.push_back(spf);

    // Right now there's nothing constraining that the object is in front of the
    // camera
    // --> A reflected solution with the object behind the camera might have
    // lower error Hack to get around this: first solve for the pose using
    // coordinate descent which DOES enforce that the object is in front of the
    // camera, then use this solution as a prior in a full LM optimization.
    StructureResult cd_result =
      optimizeStructureFromProjectionCoordinateDescent(
        normalized_coords, model, weights);

    Eigen::Matrix<double, 6, 1> prior_noise;
    prior_noise << 10, 10, 10, 0.5, 0.5,
      0.5; // ordering is q,p so there's basically no prior on q here
    auto prior_model = gtsam::noiseModel::Diagonal::Sigmas(prior_noise);

    gtsam::Pose3 prior_pose(gtsam::Rot3(cd_result.R), cd_result.t);
    gtsam::PriorFactor<gtsam::Pose3> prior(sym::O(0), prior_pose, prior_model);
    graph.push_back(prior);

    gtsam::LevenbergMarquardtParams lm_params;
    // lm_params.setVerbosityLM("SUMMARY");
    // lm_params.setVerbosityLM("DAMPED");
    lm_params.diagonalDamping = true;
    // lm_params.maxIterations = 20;

    values.insert(sym::O(0), prior_pose);

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);

    gtsam::Values lm_result = optimizer.optimize();

    result.R = lm_result.at<gtsam::Pose3>(sym::O(0)).rotation().matrix();
    result.t = lm_result.at<gtsam::Pose3>(sym::O(0)).translation().vector();
    if (k > 0)
        result.C = lm_result.at<gtsam::Vector>(sym::C(0));

    result.Z = Eigen::VectorXd(m);
    for (size_t i = 0; i < m; ++i) {
        result.Z(i) = lm_result.at<double>(sym::Z(i));
    }

    // Compute "fval" error for comparison
    // Eigen::MatrixXd S = mu;
    // if (k > 0) S += composeShape(pc, result.C);

    // Eigen::MatrixXd Sp = normalized_coords * result.Z.asDiagonal();
    // Eigen::MatrixXd St = Sp.colwise() - result.t;
    // double fval = ((St -
    // result.R*S)*weights.array().sqrt().matrix().asDiagonal()).squaredNorm();
    // if (k > 0) {
    //     fval += lambda * result.C.squaredNorm();
    // }
    // std::cout << "Optimization final fval = " << fval << std::endl;

    if (compute_covariance) {
        gtsam::Marginals marginals(graph, lm_result);
        result.Z_covariance = Eigen::VectorXd(m);
        for (size_t i = 0; i < m; ++i) {
            result.Z_covariance(i) =
              marginals.marginalCovariance(sym::Z(i))(0, 0);
        }
    }

    return result;
}

StructureResult
optimizeStructureFromProjectionCoordinateDescent(
  const Eigen::MatrixXd& normalized_coords,
  const ObjectModelBasis& model,
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

        Eigen::Vector3d reflect(
          1, 1, sgn((svd.matrixU() * svd.matrixV().transpose()).determinant()));

        R = svd.matrixU() * reflect.asDiagonal() * svd.matrixV().transpose();

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

StructureResult
optimizeStructureCoordinateDescent(const Eigen::MatrixXd& points,
                                   const ObjectModelBasis& model,
                                   const Eigen::VectorXd& weights)
{
    // cout << "Optimizing structure." << endl;
    // std::cout << "points = " << endl << points << endl;

    // cout << "weights:\n" << weights.transpose() << endl;

    // cout << endl << "model mu:" << endl << model.mu << endl;

    double lambda = 10;
    double tolerance = 1e-3;
    // bool verbose = true;

    size_t max_iters = 10; // a rough estimate here for initialization only...

    // Eigen::MatrixXd D = weights.asDiagonal();

    Eigen::MatrixXd mu = centralize(model.mu);
    Eigen::MatrixXd pc = centralize(model.pc);

    StructureResult result;

    // initialization
    Eigen::MatrixXd S = mu;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    double weight_sum = weights.sum(); // assume > 0

    Eigen::MatrixXd PminRS = points - R * S;

    Eigen::MatrixXd PminRSD = PminRS * weights.asDiagonal();
    Eigen::Vector3d t = PminRSD.rowwise().sum() / weight_sum;
    Eigen::MatrixXd St = points.colwise() - t;

    // there are size(pc, 1) / 3 basis elements
    Eigen::VectorXd C = Eigen::VectorXd::Zero(pc.rows() / 3, 1);

    double fval = std::numeric_limits<double>::infinity();

    for (size_t iter = 0; iter < max_iters; ++iter) {
        // update t
        PminRS = (points - R * S);
        PminRSD = PminRS * weights.asDiagonal();
        t = PminRSD.rowwise().sum() / weight_sum;

        // update R
        St = points.colwise() - t;

        Eigen::MatrixXd StDST = St * weights.asDiagonal() * S.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          StDST, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::Vector3d reflect(
          1, 1, sgn((svd.matrixU() * svd.matrixV().transpose()).determinant()));

        R = svd.matrixU() * reflect.asDiagonal() * svd.matrixV().transpose();

        // update C
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

    // cout << "Optimization result COORD DESCENT -- \nR = " << endl << result.R
    // << "\nt = " << result.t.transpose() << endl;

    result.R = R;
    result.C = C;
    result.t = t;

    return result;
}

} // namespace geometry