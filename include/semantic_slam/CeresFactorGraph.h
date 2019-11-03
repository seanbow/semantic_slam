#pragma once

#include "semantic_slam/Common.h"

#include <ceres/ceres.h>

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/CeresFactor.h"

class CeresFactorGraph
{
public:
    CeresFactorGraph();

    void addNode(CeresNodePtr node);

    void addFactor(CeresFactorPtr factor);

    bool solve();

    std::vector<gtsam::Key> keys();

    const ceres::Problem& problem() const { return *problem_; }

private:
    boost::shared_ptr<ceres::Problem> problem_;

    std::vector<CeresNodePtr> nodes_;
    std::vector<CeresFactorPtr> factors_;

    ceres::Solver::Options solver_options_;
};

CeresFactorGraph::CeresFactorGraph()
{
    problem_ = boost::make_shared<ceres::Problem>(); 

    // solver_options_.linear_solver_type = ceres::DENSE_SCHUR; // todo
    // solver_options_.minimizer_progress_to_stdout = true;
}

bool CeresFactorGraph::solve()
{
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, problem_.get(), &summary);

    // std::cout << summary.FullReport() << std::endl;
}

void CeresFactorGraph::addNode(CeresNodePtr node)
{
    nodes_.push_back(node);
    node->addToProblem(problem_);
}

void CeresFactorGraph::addFactor(CeresFactorPtr factor)
{
    factors_.push_back(factor);
    factor->addToProblem(problem_);
}

std::vector<gtsam::Key> 
CeresFactorGraph::keys() {
    std::vector<gtsam::Key> result;
    result.reserve(nodes_.size());
    for (auto& node : nodes_) {
        result.push_back(node->key());
    }
    return result;
}