#pragma once

#include "semantic_slam/DataAssociator.h"

class MLDataAssociator : public DataAssociator {
public:
	MLDataAssociator() { }
	MLDataAssociator(ObjectParams params) : DataAssociator(params) { }

	Eigen::MatrixXd computeConstraintWeights(const Eigen::MatrixXd& likelihoods);

};
