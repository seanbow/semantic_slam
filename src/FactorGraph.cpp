
#include "semantic_slam/FactorGraph.h"

#include <rosfmt/rosfmt.h>

FactorGraph::FactorGraph()
    : modified_(false)
{
    problem_ = boost::make_shared<ceres::Problem>(); 

    // solver_options_.linear_solver_type = ceres::DENSE_SCHUR; // todo
    // solver_options_.linear_solver_type = ceres::DENSE_QR; // todo
    // solver_options_.minimizer_progress_to_stdout = true;

    // set covariance options if needed...
    // covariance_options_....?
    covariance_ = boost::make_shared<ceres::Covariance>(covariance_options_);
}

bool FactorGraph::setNodeConstant(CeresNodePtr node)
{
    for (auto& block : node->parameter_blocks()) {
        problem_->SetParameterBlockConstant(block);
    }
    return true;
}

bool FactorGraph::solve(bool verbose)
{
    ceres::Solver::Summary summary;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ceres::Solve(solver_options_, problem_.get(), &summary);
        modified_ = false;
    }

    if (verbose)
        std::cout << summary.FullReport() << std::endl;

    return true;
}

void FactorGraph::addNode(CeresNodePtr node)
{
    if (nodes_.find(node->key()) != nodes_.end()) {
        throw std::runtime_error(
                fmt::format("Tried to add already existing node with symbol {} to graph",
                            DefaultKeyFormatter(node->key())));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    nodes_[node->key()] = node;
    node->addToProblem(problem_);
}

void FactorGraph::addFactor(CeresFactorPtr factor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    factors_.push_back(factor);
    factor->addToProblem(problem_);
}

void FactorGraph::addFactors(std::vector<CeresFactorPtr> factors)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& factor : factors) {
        factors_.push_back(factor);
        factor->addToProblem(problem_);
    }
}

std::vector<Key> 
FactorGraph::keys() {
    std::vector<Key> result;
    result.reserve(nodes_.size());
    for (auto& node : nodes_) {
        result.push_back(node.first);
    }
    return result;
}


bool FactorGraph::computeMarginalCovariance(const std::vector<Key>& keys)
{
    // Retrieve nodes and call other covariance computation function
    std::vector<CeresNodePtr> nodes;
    for (auto& key : keys) {
        auto node = getNode(key);
        if (node) nodes.push_back(node);
    }
    return computeMarginalCovariance(nodes);
}

bool FactorGraph::computeMarginalCovariance(const std::vector<CeresNodePtr>& nodes)
{
    std::vector<std::pair<const double*, const double*>> cov_blocks;

    for (size_t node_i = 0; node_i < nodes.size(); ++node_i) {
        for (size_t block_i = 0; block_i < nodes[node_i]->parameter_blocks().size(); ++block_i) {

            for (size_t node_j = node_i; node_j < nodes.size(); ++node_j) {
                // If node_j == node_i, want block_j index to start at block_i. 
                // Else, want it to start at 0.
                size_t block_j = node_j == node_i ? block_i : 0;
                for (; block_j < nodes[node_j]->parameter_blocks().size(); ++block_j) {
                    cov_blocks.push_back(std::make_pair(nodes[node_i]->parameter_blocks()[block_i],
                                                        nodes[node_j]->parameter_blocks()[block_j]));
                }
            }
        }
    }

    return covariance_->Compute(cov_blocks, problem_.get());
}


Eigen::MatrixXd FactorGraph::getMarginalCovariance(const Key& key1, const Key& key2) const 
{
    auto node1 = getNode(key1);
    auto node2 = getNode(key2);
    if (!node1 || !node2) {
        throw std::runtime_error("Error: tried to get the covariance of a node not in the FactorGraph");
    }

    return getMarginalCovariance(node1, node2);
}

Eigen::MatrixXd FactorGraph::getMarginalCovariance(CeresNodePtr node1, CeresNodePtr node2) const
{
    // Determine the size of the final matrix
    // node2 is allowed to be null here!
    size_t dim = node1->local_dim();
    if (node2) dim += node2->local_dim();

    using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // RowMajor to interface with ceres
    RowMajorMatrixXd C(dim, dim);

    // Big data buffer for ceres to write into
    // TODO do this better
    // std::vector memory is guaranteed to be contiguous. use it for RAII purposes
    std::vector<double> data_buffer_vec(dim*dim, 0.0);
    double* buf = &data_buffer_vec[0];

    // need to streamline / standardize how data is stored in these CeresNode objects...

    // Iterate over all parameter blocks
    // Start with node1 <-> node1 pairs, then node1 <-> node2, then node2 <-> node2.
    // no need for node2 <-> node1 (symmetry)
    size_t block_index_i = 0;
    size_t block_index_j = 0;
    for (size_t i = 0; i < node1->parameter_blocks().size(); ++i) {
        size_t i_dim = node1->parameter_block_local_sizes()[i];

        // node1 <-> node1
        block_index_j = 0;
        for (size_t j = 0; j < node1->parameter_blocks().size(); ++j) {
            size_t j_dim = node1->parameter_block_local_sizes()[j];

            // j < i is below the matrix diagonal, don't need to compute it
            if (j >= i) {
                covariance_->GetCovarianceBlockInTangentSpace(node1->parameter_blocks()[i], 
                                                              node1->parameter_blocks()[j], 
                                                              buf);

                C.block(block_index_i, block_index_j, i_dim, j_dim) = Eigen::Map<RowMajorMatrixXd>(buf, i_dim, j_dim);
            }

            block_index_j += j_dim;
        }

        // node1 <-> node2
        if (node2) {
            for (size_t j = 0; j < node2->parameter_blocks().size(); ++j) {
                size_t j_dim = node2->parameter_block_local_sizes()[j];

                covariance_->GetCovarianceBlockInTangentSpace(node1->parameter_blocks()[i], 
                                                              node2->parameter_blocks()[j], 
                                                              buf);
                C.block(block_index_i, block_index_j, i_dim, j_dim) = Eigen::Map<RowMajorMatrixXd>(buf, i_dim, j_dim);

                block_index_j += j_dim;
            }
        }

        block_index_i += i_dim;
    }

    // Finally, node2 <-> node2
    if (node2) {
        for (size_t i = 0; i < node2->parameter_blocks().size(); ++i) {
            size_t i_dim = node2->parameter_block_local_sizes()[i];

            // Start "j" index here will be the total dimension of the node1 blocks
            block_index_j = node1->local_dim();
            for (size_t j = 0; j < node2->parameter_blocks().size(); ++j) {
                size_t j_dim = node2->parameter_block_local_sizes()[j];

                covariance_->GetCovarianceBlockInTangentSpace(node2->parameter_blocks()[i], 
                                                              node2->parameter_blocks()[j], 
                                                              buf);
                C.block(block_index_i, block_index_j, i_dim, j_dim) = Eigen::Map<RowMajorMatrixXd>(buf, i_dim, j_dim);

                block_index_j += j_dim;
            }

            block_index_i += i_dim;
        }
    }

    // ROS_INFO_STREAM("Covariance = \n" << C );

    // Have only filled in the block-upper-triangular portion, let Eigen fill in the rest
    return C.selfadjointView<Eigen::Upper>();
}

CeresNodePtr
FactorGraph::findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0) return nullptr;

    ros::Time last_time(0);
    CeresNodePtr node = nullptr;

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr) continue;

        if (!key_node.second->time()) continue;

        if (key_node.second->time() > last_time && key_node.second->time() <= time) {
            last_time = *key_node.second->time();
            node = key_node.second;
        }
    }

    return node;
}


CeresNodePtr
FactorGraph::findFirstNodeAfterTime(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0) return nullptr;

    ros::Time first_time = ros::TIME_MAX;
    CeresNodePtr node = nullptr;

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr) continue;

        if (!key_node.second->time()) continue;

        if (key_node.second->time() <= first_time && key_node.second->time() >= time) {
            first_time = *key_node.second->time();
            node = key_node.second;
        }
    }

    return node;
}


CeresNodePtr
FactorGraph::findNearestNode(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0) return nullptr;

    ros::Duration shortest_duration = ros::DURATION_MAX;
    CeresNodePtr node = nullptr;

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr) continue;

        if (!key_node.second->time()) continue;

        if (abs_duration(time - *key_node.second->time()) <= shortest_duration) {
            shortest_duration = abs_duration(time - *key_node.second->time());
            node = key_node.second;
        }
    }

    return node;
}

