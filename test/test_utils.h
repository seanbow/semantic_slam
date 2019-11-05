#include "semantic_slam/SE3Node.h"
#include <ceres/ceres.h>

#include <gtest/gtest.h>

#include "semantic_slam/CeresFactorGraph.h"
#include "semantic_slam/CeresSE3PriorFactor.h"
#include "semantic_slam/CeresBetweenFactor.h"

#include <fstream>
#include <string>
#include <unordered_map>

using namespace std::string_literals;


// g2o file input code here is modified from ceres source code examples 

SE3NodePtr readVertex(std::ifstream* infile)
{
    int id;
    Eigen::Vector3d t;
    math::Quaternion q;
    *infile >> id >> t(0) >> t(1) >> t(2) >> q(0) >> q(1) >> q(2) >> q(3);

    q = math::quat_inv(q);

    // std::cout << "Read vertex " << id << std::endl;

    SE3NodePtr node = boost::make_shared<SE3Node>(gtsam::Symbol('x', id));
    node->pose().translation() = t;
    node->pose().rotation() = q;
    node->pose().rotation().normalize();

    return node;
}

CeresFactorPtr readFactor(std::ifstream* infile, const std::unordered_map<gtsam::Key, SE3NodePtr>& nodes)
{
    int id1, id2;
    math::Quaternion q;
    Eigen::Vector3d p;
    Eigen::MatrixXd information = Eigen::MatrixXd::Zero(6,6);

    *infile >> id1 >> id2;

    *infile >> p(0) >> p(1) >> p(2) >> q(0) >> q(1) >> q(2) >> q(3);
    q.normalize();
    q = math::quat_inv(q);
    Pose3 between(q,p);

    for (int i = 0; i < 6 && infile->good(); ++i) {
        for (int j = i; j < 6 && infile->good(); ++j) {
            *infile >> information(i,j);
            if (i != j) information(j,i) = information(i,j);
        }
    }

    // g2o has information in [p, q] order, need to swap to [q,p]:

    Eigen::Matrix<double,6,6> reorder;

    reorder.block<3,3>(0,0) = information.block<3,3>(3,3);
    reorder.block<3,3>(0,3) = information.block<3,3>(0,3);
    reorder.block<3,3>(3,0) = information.block<3,3>(3,0);
    reorder.block<3,3>(3,3) = information.block<3,3>(0,0);

    Eigen::Matrix<double,6,6> cov = reorder.partialPivLu().inverse();


    // std::cout << " Read edge " << id1 << " -> " << id2 << std::endl;

    gtsam::Symbol sym1('x', id1);
    gtsam::Symbol sym2('x', id2);
    CeresBetweenFactorPtr factor= boost::make_shared<CeresBetweenFactor>(nodes.at(sym1.key()), 
                                                                        nodes.at(sym2.key()),
                                                                        between, 
                                                                        cov);

    return boost::static_pointer_cast<CeresFactor>(factor);
}

bool readG2oFile(const std::string& filename,
                   std::unordered_map<gtsam::Key, SE3NodePtr>& nodes,
                   std::vector<CeresFactorPtr>& factors)
{
    nodes.clear();
    factors.clear();

    std::ifstream infile(filename.c_str());

    std::string data_type;
    while (infile.good()) {
        // Check if we have a node or a constraint
        infile >> data_type;
        if (data_type == "VERTEX_SE3:QUAT"s) {
            SE3NodePtr node = readVertex(&infile);
            nodes[node->key()] = node;
        } else {
            CeresFactorPtr factor = readFactor(&infile, nodes);
            factors.push_back(factor);
        }

        // Clear any trailing whitespace
        infile >> std::ws;
    }
}

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool outputPoses(const std::string& filename, const std::unordered_map<gtsam::Key, SE3NodePtr>& poses) {
    std::fstream outfile;
    outfile.open(filename.c_str(), std::istream::out);
    if (!outfile) {
        std::cerr << "Error opening the file: " << filename;
        return false;
    }

    // sort in index order
    std::vector<SE3NodePtr> poses_vec;
    for (auto& key_node : poses) {
        poses_vec.push_back(key_node.second);
    }

    std::sort(poses_vec.begin(), poses_vec.end(), [](const SE3NodePtr& a, const SE3NodePtr& b) {
        return a->index() < b->index();
    });

    for (auto& pose : poses_vec) {
        const Eigen::Vector3d& p = pose->translation();
        const math::Quaternion& q = pose->rotation();

        outfile << pose->index() << " " << p(0) << " " 
                << p(1) << " "  << p(2) << " "  
                << q(0) << " "  << q(1) << " "  
                << q(2) << " "  << q(3) << '\n';
    }

    return true;
}