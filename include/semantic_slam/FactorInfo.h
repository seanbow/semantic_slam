#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include "semantic_slam/registry.h"

/**
 * Class that stores a factor graph factor along with some meta-information
 */
class FactorInfo {
public:
    FactorInfo()
        : in_graph(false) { }

    FactorInfo(const boost::shared_ptr<gtsam::NonlinearFactor>& fac)
        : in_graph(false),
          factor_(fac) { }

    FactorInfo(FactorType type, int tag)
        : in_graph(false),
          type_(type),
          tag_(tag) { }

    FactorInfo(FactorType type, int tag, const boost::shared_ptr<gtsam::NonlinearFactor>& fac)
        : in_graph(false),
          factor_(fac),
          type_(type),
          tag_(tag) { }

    FactorInfo(FactorType type, const boost::shared_ptr<gtsam::NonlinearFactor>& fac)
        : in_graph(false),
          factor_(fac),
          type_(type) { }

    boost::shared_ptr<gtsam::NonlinearFactor> factor() { return factor_; }

    template <typename FactorT>
    boost::shared_ptr<FactorT> factor() { return boost::static_pointer_cast<FactorT>(factor_); }

    FactorType type() const { return type_; }

    int tag() const { return tag_; }

    size_t index; //< Index within a graph or optimization structure e.g. gtsam::ISAM2

    bool in_graph;

    using Ptr = boost::shared_ptr<FactorInfo>;

private:
    boost::shared_ptr<gtsam::NonlinearFactor> factor_;

    FactorType type_;

    int tag_;
};

using FactorInfoPtr = boost::shared_ptr<FactorInfo>;

