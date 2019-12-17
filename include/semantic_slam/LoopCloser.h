#pragma once

#include "semantic_slam/FactorGraph.h"
#include <boost/shared_ptr.hpp>
#include <ceres/ceres.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

class SemanticMapper;

class LoopCloser
{
  public:
    LoopCloser(SemanticMapper* mapper);

    void startLoopClosing(boost::shared_ptr<FactorGraph> graph,
                          int loop_closing_kf_index);

    bool containsNode(Key key);

    bool updateLoopInMapper();

    bool running();

  private:
    std::thread thread_;
    std::mutex mutex_;

    SemanticMapper* mapper_;

    boost::shared_ptr<FactorGraph> current_graph_;
    int loop_index_;
    bool solve_succeeded_;

    std::atomic<bool> running_;
    std::chrono::high_resolution_clock::time_point time_start_;
    std::chrono::high_resolution_clock::time_point time_end_;

    void optimizeCurrentGraph();
};
