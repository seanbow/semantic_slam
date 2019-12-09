#pragma once

#include <gtsam/inference/Symbol.h>

/**
 * Class that corresponds to a node in the factor graph with some
 * meta-information
 */
class NodeInfo
{
  public:
    NodeInfo()
      : in_graph(false)
    {}

    NodeInfo(gtsam::Symbol symbol,
             boost::optional<ros::Time> time = boost::none)
      : in_graph(false)
      , symbol_(symbol)
      , time_(time)
    {}

    gtsam::Symbol symbol() const { return symbol_; }
    gtsam::Key key() const { return symbol_; /* implicit conversion */ }

    unsigned char chr() const { return symbol_.chr(); }
    size_t index() const { return symbol_.index(); }

    boost::optional<ros::Time> time() const { return time_; }

    bool in_graph;

    using Ptr = boost::shared_ptr<NodeInfo>;
    using ConstPtr = boost::shared_ptr<const NodeInfo>;

    static NodeInfo::Ptr Create() { return boost::make_shared<NodeInfo>(); }
    static NodeInfo::Ptr Create(gtsam::Symbol symbol,
                                boost::optional<ros::Time> time = boost::none)
    {
        return boost::make_shared<NodeInfo>(symbol, time);
    }

  private:
    gtsam::Symbol symbol_;

    boost::optional<ros::Time> time_;
};

using NodeInfoPtr = NodeInfo::Ptr;
using NodeInfoConstPtr = NodeInfo::ConstPtr;
