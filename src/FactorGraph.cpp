
#include "semantic_slam/FactorGraph.h"

#include <rosfmt/rosfmt.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

FactorGraph::FactorGraph()
  : modified_(false)
{
    ceres::Problem::Options problem_options;
    problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.enable_fast_removal = true;
    problem_ = boost::make_shared<ceres::Problem>(problem_options);

    // set covariance options if needed...
    // covariance_options_....?
    covariance_options_.apply_loss_function = true;
    covariance_options_.num_threads = 4;
    // covariance_options_.algorithm_type = ceres::SPARSE_CHOLESKY;
    covariance_ = boost::make_shared<ceres::Covariance>(covariance_options_);
}

void
FactorGraph::setSolverOptions(ceres::Solver::Options solver_options)
{
    solver_options_ = solver_options;
}

void
FactorGraph::setNumThreads(int n_threads)
{
    solver_options_.num_threads = n_threads;
    covariance_options_.num_threads = n_threads;
}

bool
FactorGraph::setNodeConstant(CeresNodePtr node)
{
    for (auto& block : node->parameter_blocks()) {
        problem_->SetParameterBlockConstant(block);
    }
    return true;
}

bool
FactorGraph::setNodeVariable(CeresNodePtr node)
{
    for (auto& block : node->parameter_blocks()) {
        problem_->SetParameterBlockVariable(block);
    }
    return true;
}

bool
FactorGraph::isNodeConstant(CeresNodePtr node) const
{
    // assume that the user is not interfacing with the ceres::Problem
    // directly... i.e. assume that one of the node's parameter blocks is
    // constant iff they all are.
    return problem_->IsParameterBlockConstant(node->parameter_blocks()[0]);
}

boost::shared_ptr<FactorGraph>
FactorGraph::clone() const
{
    auto new_graph = util::allocate_aligned<FactorGraph>();

    // Create the set of new nodes over which we'll be operating
    std::unordered_map<Key, CeresNodePtr> new_nodes;
    for (const auto& node : nodes_) {
        auto new_node = node.second->clone();

        new_nodes.emplace(node.first, new_node);
        new_graph->addNode(new_node);

        if (isNodeConstant(node.second)) {
            new_graph->setNodeConstant(new_node);
        }
    }

    // Need to iterate over each factor, clone it, and set it to operate on
    // the new nodes
    for (const auto& factor : factors_) {
        auto new_fac = factor->clone();

        // new_fac contains all the correct measurement info etc but is
        // currently set with all NULL nodes. collect the set of nodes based on
        // keys from the old factor
        std::vector<CeresNodePtr> new_factor_nodes;
        for (const auto& old_node : factor->nodes()) {
            new_factor_nodes.push_back(new_nodes[old_node->key()]);
        }

        new_fac->setNodes(new_factor_nodes);
        new_graph->addFactor(new_fac);
    }

    new_graph->setSolverOptions(solver_options_);

    // copying elimination ordering is not yet supported
    new_graph->solver_options().linear_solver_ordering = nullptr;

    return new_graph;
}

bool
FactorGraph::solve(bool verbose)
{
    ceres::Solver::Summary summary;

    solver_options_.minimizer_progress_to_stdout = verbose;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ceres::Solve(solver_options_, problem_.get(), &summary);
        modified_ = false;
    }

    if (verbose)
        std::cout << summary.FullReport() << std::endl;

    if (verbose) {
        std::cout << "Linear solver ordering sizes: \n";
        for (auto& sz : summary.linear_solver_ordering_used) {
            std::cout << sz << " ";
        }
        std::cout << std::endl;

        std::cout << "Schur structure detected: "
                  << summary.schur_structure_given << std::endl;
        std::cout << "Schur structure used: " << summary.schur_structure_used
                  << std::endl;
    }

    return summary.IsSolutionUsable();
    // return summary.termination_type == ceres::CONVERGENCE;
}

void
FactorGraph::addNode(CeresNodePtr node)
{
    if (!node)
        return;
    if (nodes_.find(node->key()) != nodes_.end()) {
        throw std::runtime_error(fmt::format(
          "Tried to add already existing node with symbol {} to graph",
          DefaultKeyFormatter(node->key())));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    nodes_[node->key()] = node;
    node->addToProblem(problem_);
    modified_ = true;
}

void
FactorGraph::addFactor(CeresFactorPtr factor)
{
    if (!factor)
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    factors_.push_back(factor);
    factor->addToProblem(problem_);
    modified_ = true;
}

void
FactorGraph::addFactors(std::vector<CeresFactorPtr> factors)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& factor : factors) {
        if (!factor)
            continue;
        factors_.push_back(factor);
        factor->addToProblem(problem_);
    }
    modified_ = true;
}

void
FactorGraph::removeNode(CeresNodePtr node)
{
    auto it = nodes_.find(node->key());
    if (it != nodes_.end()) {
        nodes_.erase(it);
    }

    modified_ = true;

    // bool found = false;
    // for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    //     if (node->key() == it->second->key()) {
    //         nodes_.erase(it);
    //         found = true;
    //         break;
    //     }
    // }

    // node->removeFromProblem(problem_);

    // modified_ = true;
}

void
FactorGraph::removeFactor(CeresFactorPtr factor)
{
    bool found = false;
    for (auto it = factors_.begin(); it != factors_.end(); ++it) {
        if (it->get() == factor.get()) {
            factors_.erase(it);
            found = true;
            break;
        }
    }

    if (found) {
        factor->removeFromProblem(problem_);
        modified_ = true;
    }
}

std::vector<Key>
FactorGraph::keys()
{
    std::vector<Key> result;
    result.reserve(nodes_.size());
    for (auto& node : nodes_) {
        result.push_back(node.first);
    }
    return result;
}

bool
FactorGraph::containsFactor(CeresFactorPtr factor)
{
    auto fac_it = std::find(factors_.begin(), factors_.end(), factor);

    if (fac_it != factors_.end()) {
        return true;
    } else {
        return false;
    }
}

bool
FactorGraph::computeMarginalCovariance(const std::vector<Key>& keys)
{
    // Retrieve nodes and call other covariance computation function
    std::vector<CeresNodePtr> nodes;
    for (auto& key : keys) {
        auto node = getNode(key);
        if (node)
            nodes.push_back(node);
    }
    return computeMarginalCovariance(nodes);
}

bool
FactorGraph::computeMarginalCovariance(const std::vector<CeresNodePtr>& nodes)
{
    std::vector<const double*> blocks;
    for (const CeresNodePtr& node : nodes) {
        blocks.insert(blocks.end(),
                      node->parameter_blocks().begin(),
                      node->parameter_blocks().end());
    }

    std::vector<std::pair<const double*, const double*>> block_pairs =
      produceAllPairs(blocks);

    return covariance_->Compute(block_pairs, problem_.get());

    // std::vector<std::pair<const double*, const double*>> cov_blocks;

    // for (size_t node_i = 0; node_i < nodes.size(); ++node_i) {
    //     for (size_t block_i = 0; block_i <
    //     nodes[node_i]->parameter_blocks().size(); ++block_i) {

    //         for (size_t node_j = node_i; node_j < nodes.size(); ++node_j) {
    //             // If node_j == node_i, want block_j index to start at
    //             block_i.
    //             // Else, want it to start at 0.
    //             size_t block_j = node_j == node_i ? block_i : 0;
    //             for (; block_j < nodes[node_j]->parameter_blocks().size();
    //             ++block_j) {
    //                 cov_blocks.push_back(std::make_pair(nodes[node_i]->parameter_blocks()[block_i],
    //                                                     nodes[node_j]->parameter_blocks()[block_j]));
    //             }
    //         }
    //     }
    // // }

    // return covariance_->Compute(cov_blocks, problem_.get());
}

Eigen::MatrixXd
FactorGraph::getMarginalCovariance(const Key& key) const
{
    auto node = getNode(key);
    if (!node)
        throw std::runtime_error("Error: tried to get the covariance of a node "
                                 "not in the FactorGraph");
    return getMarginalCovariance({ node });
}

Eigen::MatrixXd
FactorGraph::getMarginalCovariance(const Key& key1, const Key& key2) const
{
    auto node1 = getNode(key1);
    auto node2 = getNode(key2);
    if (!node1 || !node2) {
        throw std::runtime_error("Error: tried to get the covariance of a node "
                                 "not in the FactorGraph");
    }

    return getMarginalCovariance({ node1, node2 });
}

Eigen::MatrixXd
FactorGraph::getMarginalCovariance(const std::vector<CeresNodePtr>& nodes) const
{
    // Collect pointers to parameter blocks and their sizes
    // TODO need to streamline / standardize how data is stored in these
    // CeresNode objects...
    std::vector<double*> parameter_blocks;
    std::vector<size_t> parameter_block_sizes;
    size_t full_dim = 0;
    size_t max_block_size = 0;

    for (auto& node : nodes) {
        for (int i = 0; i < node->parameter_blocks().size(); ++i) {
            parameter_blocks.push_back(node->parameter_blocks()[i]);
            parameter_block_sizes.push_back(
              node->parameter_block_local_sizes()[i]);

            full_dim += node->parameter_block_local_sizes()[i];
            max_block_size =
              std::max(max_block_size, node->parameter_block_local_sizes()[i]);
        }
    }

    using RowMajorMatrixXd =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::MatrixXd C(full_dim, full_dim);

    // data buffer for ceres to write each block into...
    // std::vector<> memory is guaranteed to be contiguous, use it for RAII
    // purposes
    std::vector<double> data_buffer_vec(max_block_size * max_block_size, 0.0);
    double* buf = &data_buffer_vec[0];

    // Begin filling in the covariance matrix
    size_t index_i = 0;
    size_t index_j = 0;
    for (int i = 0; i < parameter_blocks.size(); ++i) {
        for (int j = i; j < parameter_blocks.size(); ++j) {
            covariance_->GetCovarianceBlockInTangentSpace(
              parameter_blocks[i], parameter_blocks[j], buf);

            C.block(index_i,
                    index_j,
                    parameter_block_sizes[i],
                    parameter_block_sizes[j]) =
              Eigen::Map<RowMajorMatrixXd>(
                buf, parameter_block_sizes[i], parameter_block_sizes[j]);

            index_j += parameter_block_sizes[j];
        }

        index_i += parameter_block_sizes[i];
        index_j = index_i;
    }

    return C.selfadjointView<Eigen::Upper>();
}

// Eigen::MatrixXd FactorGraph::getMarginalCovariance(CeresNodePtr node1,
// CeresNodePtr node2) const
// {
//     // Determine the size of the final matrix
//     // node2 is allowed to be null here!
//     size_t dim = node1->local_dim();
//     if (node2) dim += node2->local_dim();

//     using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic,
//     Eigen::Dynamic, Eigen::RowMajor>;

//     // RowMajor to interface with ceres
//     RowMajorMatrixXd C(dim, dim);

//     // Big data buffer for ceres to write into
//     // TODO do this better
//     // std::vector memory is guaranteed to be contiguous. use it for RAII
//     purposes std::vector<double> data_buffer_vec(dim*dim, 0.0); double* buf =
//     &data_buffer_vec[0];

//     // need to streamline / standardize how data is stored in these CeresNode
//     objects...

//     // Iterate over all parameter blocks
//     // Start with node1 <-> node1 pairs, then node1 <-> node2, then node2 <->
//     node2.
//     // no need for node2 <-> node1 (symmetry)
//     size_t block_index_i = 0;
//     size_t block_index_j = 0;
//     for (size_t i = 0; i < node1->parameter_blocks().size(); ++i) {
//         size_t i_dim = node1->parameter_block_local_sizes()[i];

//         // node1 <-> node1
//         block_index_j = 0;
//         for (size_t j = 0; j < node1->parameter_blocks().size(); ++j) {
//             size_t j_dim = node1->parameter_block_local_sizes()[j];

//             // j < i is below the matrix diagonal, don't need to compute it
//             if (j >= i) {
//                 covariance_->GetCovarianceBlockInTangentSpace(node1->parameter_blocks()[i],
//                                                               node1->parameter_blocks()[j],
//                                                               buf);

//                 C.block(block_index_i, block_index_j, i_dim, j_dim) =
//                 Eigen::Map<RowMajorMatrixXd>(buf, i_dim, j_dim);
//             }

//             block_index_j += j_dim;
//         }

//         // node1 <-> node2
//         if (node2) {
//             for (size_t j = 0; j < node2->parameter_blocks().size(); ++j) {
//                 size_t j_dim = node2->parameter_block_local_sizes()[j];

//                 covariance_->GetCovarianceBlockInTangentSpace(node1->parameter_blocks()[i],
//                                                               node2->parameter_blocks()[j],
//                                                               buf);
//                 C.block(block_index_i, block_index_j, i_dim, j_dim) =
//                 Eigen::Map<RowMajorMatrixXd>(buf, i_dim, j_dim);

//                 block_index_j += j_dim;
//             }
//         }

//         block_index_i += i_dim;
//     }

//     // Finally, node2 <-> node2
//     if (node2) {
//         for (size_t i = 0; i < node2->parameter_blocks().size(); ++i) {
//             size_t i_dim = node2->parameter_block_local_sizes()[i];

//             // Start "j" index here will be the total dimension of the node1
//             blocks block_index_j = node1->local_dim(); for (size_t j = 0; j <
//             node2->parameter_blocks().size(); ++j) {
//                 size_t j_dim = node2->parameter_block_local_sizes()[j];

//                 covariance_->GetCovarianceBlockInTangentSpace(node2->parameter_blocks()[i],
//                                                               node2->parameter_blocks()[j],
//                                                               buf);
//                 C.block(block_index_i, block_index_j, i_dim, j_dim) =
//                 Eigen::Map<RowMajorMatrixXd>(buf, i_dim, j_dim);

//                 block_index_j += j_dim;
//             }

//             block_index_i += i_dim;
//         }
//     }

//     // ROS_INFO_STREAM("Covariance = \n" << C );

//     // Have only filled in the block-upper-triangular portion, let Eigen fill
//     in the rest return C.selfadjointView<Eigen::Upper>();
// }

CeresNodePtr
FactorGraph::findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0)
        return nullptr;

    ros::Time last_time(0);
    CeresNodePtr node = nullptr;

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr)
            continue;

        if (!key_node.second->time())
            continue;

        if (key_node.second->time() > last_time &&
            key_node.second->time() <= time) {
            last_time = *key_node.second->time();
            node = key_node.second;
        }
    }

    return node;
}

CeresNodePtr
FactorGraph::findFirstNodeAfterTime(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0)
        return nullptr;

    ros::Time first_time = ros::TIME_MAX;
    CeresNodePtr node = nullptr;

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr)
            continue;

        if (!key_node.second->time())
            continue;

        if (key_node.second->time() <= first_time &&
            key_node.second->time() >= time) {
            first_time = *key_node.second->time();
            node = key_node.second;
        }
    }

    return node;
}

CeresNodePtr
FactorGraph::findNearestNode(unsigned char symbol_chr, ros::Time time)
{
    if (nodes_.size() == 0)
        return nullptr;

    ros::Duration shortest_duration = ros::DURATION_MAX;
    CeresNodePtr node = nullptr;

    for (auto& key_node : nodes_) {
        if (key_node.second->chr() != symbol_chr)
            continue;

        if (!key_node.second->time())
            continue;

        if (abs_duration(time - *key_node.second->time()) <=
            shortest_duration) {
            shortest_duration = abs_duration(time - *key_node.second->time());
            node = key_node.second;
        }
    }

    return node;
}

boost::shared_ptr<gtsam::NonlinearFactorGraph>
FactorGraph::getGtsamGraph() const
{
    auto graph = util::allocate_aligned<gtsam::NonlinearFactorGraph>();

    for (auto factor : factors_) {
        // if (factor->active()) {
        //     bool good = true;
        //     for (auto& node : factor->nodes()) {
        //         if (!node->active()) {
        //             good = false;
        //             break;
        //         }
        //     }

        //     if (good) graph->push_back(factor->getGtsamFactor());
        // }

        // if (factor->active()) graph->push_back(factor->getGtsamFactor());
        if (factor->active())
            factor->addToGtsamGraph(graph);
    }

    return graph;
}

boost::shared_ptr<gtsam::Values>
FactorGraph::getGtsamValues() const
{
    auto values = util::allocate_aligned<gtsam::Values>();

    for (auto node : nodes_) {
        if (node.second->active())
            values->insert(node.first, *node.second->getGtsamValue());
    }

    return values;
}
