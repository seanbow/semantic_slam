#pragma once

#include "semantic_slam/Common.h"

#include <ceres/ceres.h>

#include "semantic_slam/CeresNode.h"
#include "semantic_slam/CeresFactor.h"

#include <unordered_map>

class CeresFactorGraph
{
public:
    CeresFactorGraph();

    void addNode(CeresNodePtr node);

    void addFactor(CeresFactorPtr factor);

    bool solve(bool verbose=false);

    std::vector<Key> keys();

    const ceres::Problem& problem() const { return *problem_; }

    bool setFirstNodeConstant();

private:
    boost::shared_ptr<ceres::Problem> problem_;

    std::unordered_map<Key, CeresNodePtr> nodes_;
    std::vector<CeresFactorPtr> factors_;

    ceres::Solver::Options solver_options_;
};

CeresFactorGraph::CeresFactorGraph()
{
    problem_ = boost::make_shared<ceres::Problem>(); 

    // solver_options_.linear_solver_type = ceres::DENSE_SCHUR; // todo
    // solver_options_.linear_solver_type = ceres::DENSE_QR; // todo
    // solver_options_.minimizer_progress_to_stdout = true;
}

bool CeresFactorGraph::setFirstNodeConstant()
{
    if (nodes_.size() == 0) return false;
    auto node = nodes_.find(Symbol('x', 0));
    if (node == nodes_.end()) {
        std::cout << "Node X0 not in nodes" << std::endl;
        return false;
    }
    for (double* block : node->second->parameter_blocks()) {
        problem_->SetParameterBlockConstant(block);
    }
    return true;
}

bool CeresFactorGraph::solve(bool verbose)
{
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options_, problem_.get(), &summary);

    if (verbose)
        std::cout << summary.FullReport() << std::endl;
}

void CeresFactorGraph::addNode(CeresNodePtr node)
{
    if (nodes_.find(node->key()) != nodes_.end()) {
        throw std::runtime_error("Node already exists in factor graph");
    }
    nodes_[node->key()] = node;

    node->addToProblem(problem_);
}

void CeresFactorGraph::addFactor(CeresFactorPtr factor)
{
    factors_.push_back(factor);
    factor->addToProblem(problem_);
}

std::vector<Key> 
CeresFactorGraph::keys() {
    std::vector<Key> result;
    result.reserve(nodes_.size());
    for (auto& node : nodes_) {
        result.push_back(node.first);
    }
    return result;
}