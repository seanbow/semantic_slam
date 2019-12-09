
#include "semantic_slam/MLDataAssociator.h"
#include "semantic_slam/munkres.h"

Eigen::MatrixXd
MLDataAssociator::computeConstraintWeights(const Eigen::MatrixXd& mahals)
{
    size_t m = mahals.rows();
    size_t n = mahals.cols();

    if (m == 0) {
        return Eigen::MatrixXd::Zero(m, n + 1);
    }

    // std::cout << "Mahals matrix is " << m << " by " << n << std::endl;

    // Prepare matrix for munkres algorithm
    // Pad with distance thresholds
    Eigen::MatrixXd munkres_mat(m, n + m);

    munkres_mat.block(0, 0, m, n) = mahals;
    munkres_mat.block(0, n, m, m) =
      params_.mahal_thresh_assign * Eigen::MatrixXd::Ones(m, m);

    // std::cout << "Computing weights with munkres matrix: " << std::endl <<
    // munkres_mat << std::endl;

    Eigen::MatrixXi munkres_result = Munkres().solve(munkres_mat);

    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(m, n + 1);

    // Compute the min of each row in mahals to check for new landmark
    // assignment
    Eigen::VectorXd msmt_mahal_mins;
    if (n > 0) {
        msmt_mahal_mins = mahals.rowwise().minCoeff();
    }

    // TODO check for and remove ambiguous assignments? e.g. if the winning
    // assignment j is such that mahals(i,j) is close to mahals(i,k) for some k
    // != j

    for (int64_t i = 0; i < munkres_result.rows(); ++i) {
        for (int64_t j = 0; j < munkres_result.cols(); ++j) {
            if (munkres_result(i, j) == 0) {
                // Assignment found
                // If j < the number of landmarks, it's an assignment to an
                // existing one
                if (j < n) {
                    weights(i, j) = 1;
                } else {
                    // Assignment returned in padded part of matrix, i.e.
                    // "unassigned" Check if we should initialize a new one
                    if (n == 0 ||
                        msmt_mahal_mins(i) > params_.mahal_thresh_init) {
                        weights(i, n) = 1;
                    }
                }
                break; // on to next measurement
            }
        }
    }

    // std::cout << "Munkres result is: " << std::endl << munkres_result <<
    // std::endl;

    return weights;
}