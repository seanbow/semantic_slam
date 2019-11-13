#ifndef SEMSLAM_DATAASSOCIATOR_H_
#define SEMSLAM_DATAASSOCIATOR_H_

#include "semantic_slam/keypoints/EstimatedKeypoint.h"

class DataAssociator {
public:
	virtual Eigen::MatrixXd computeConstraintWeights(const Eigen::MatrixXd& likelihoods) = 0;

	virtual void setParams(ObjectParams params) { params_ = params; }

	virtual ~DataAssociator() { }

protected:
	DataAssociator() { }
	DataAssociator(ObjectParams params) : params_(params) { }

	ObjectParams params_;
};

#endif
