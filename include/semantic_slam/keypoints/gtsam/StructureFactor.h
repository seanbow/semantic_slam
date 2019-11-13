#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/JacobianFactor.h>

#include <boost/optional.hpp>
#include <boost/make_shared.hpp>

#include <semantic_slam/keypoints/geometry.h>
#include <semantic_slam/Utils.h>

class StructureFactorTester;

namespace semslam {

using geometry::ObjectModelBasis;

class StructureFactor : public gtsam::NonlinearFactor {
private:
    typedef gtsam::NonlinearFactor Base;
    typedef StructureFactor This;
    
public:
    
    typedef boost::shared_ptr<This> shared_ptr;
    typedef shared_ptr Ptr;

    StructureFactor(gtsam::Key object_key,
                    const std::vector<gtsam::Key>& landmark_keys,
                    const gtsam::Key& coefficient_key,
                    const ObjectModelBasis& model,
                    const Eigen::VectorXd& weights,
                    double lambda=1.0);

    void setWeights(const Eigen::VectorXd& weights) { weights_ = weights; }
    
    gtsam::Vector unwhitenedError(const gtsam::Values& values,
                                  boost::optional< std::vector<gtsam::Matrix>& > = boost::none) const;

    gtsam::Vector whitenedError(const gtsam::Values& values) const;

    // *total* error
    double error(const gtsam::Values& values) const;

    boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const;

    size_t dim() const {
        return 3*m_ + k_;
    }

    /** print contents */
    void print(const std::string& s="", const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const {
        std::cout << s << "StructureFactor" << std::endl;
        Base::print("", keyFormatter);
    }

    /// @return a deep copy of this factor
    // virtual gtsam::NonlinearFactor::shared_ptr clone() const {
    //     return boost::static_pointer_cast<gtsam::NonlinearFactor>(
    //         gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    // }
                                                                                                    
private:

    gtsam::Key object_key_;
    std::vector<gtsam::Key> landmark_keys_; 
    gtsam::Key coefficient_key_;

    ObjectModelBasis model_;

    double lambda_; // regularization factor

    size_t m_, k_;

    // gtsam::noiseModel::Base::shared_ptr noise_model_;

    Eigen::VectorXd weights_;

    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Bi_;

    Eigen::MatrixXd Dobject(const gtsam::Values& values) const;
    Eigen::MatrixXd Dcoefficients(const gtsam::Values& values) const;
    Eigen::MatrixXd Dlandmark(const gtsam::Values& values, size_t landmark_index) const;

    Eigen::MatrixXd structure(const gtsam::Values& values) const;

    
    void whitenError(gtsam::Vector& e) const;

    void whitenJacobians(std::vector<gtsam::Matrix>& vec) const;

    void whitenJacobian(gtsam::Matrix& H) const;

    friend class ::StructureFactorTester;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

using StructureFactorPtr = StructureFactor::Ptr;

} // namespace semslam
