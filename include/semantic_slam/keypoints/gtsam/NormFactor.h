#pragma once

#include <gtsam/nonlinear/ExpressionFactor.h>

namespace gtsam {

/**
 * Unary factor for a norm measurement
 */
template<typename T>
class NormFactor : public NoiseModelFactor1<T>
{
  public:
    typedef NormFactor<T> This;
    typedef NoiseModelFactor1<T> Base;

    /// default constructor
    NormFactor() {}

    NormFactor(Key key, double measured, const SharedNoiseModel& model)
      : Base(model, key)
      , prior_(measured)
    {}

    Vector evaluateError(const T& x,
                         boost::optional<Matrix&> H = boost::none) const
    {
        double norm = x.norm();
        if (H) {
            *H = x.transpose() / norm;
        }
        auto err = Eigen::VectorXd(1);
        err(0) = x.norm() - prior_;
        return err;
    }

    /// @return a deep copy of this factor
    gtsam::NonlinearFactor::shared_ptr clone() const
    {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    void print(const std::string& s = "",
               const KeyFormatter& kf = DefaultKeyFormatter) const
    {
        std::cout << s << "NormFactor on " << kf(this->key()) << std::endl;
    }

  private:
    double prior_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace gtsam
