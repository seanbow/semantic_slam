#include "semantic_slam/FactorGraph.h"

#include "semantic_slam/Camera.h"
#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/CeresProjectionFactor.h"
#include "semantic_slam/CeresSE3PriorFactor.h"
#include "semantic_slam/MultiProjectionFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SmartProjectionFactor.h"
#include "semantic_slam/Symbol.h"
#include "semantic_slam/VectorNode.h"
#include "semantic_slam/quaternion_math.h"
#include <gtest/gtest.h>
#include <random>

namespace sym = symbol_shorthand;

using NodeType = VectorNode<2>;

// Set this high to turn off this gate in these tests
double reproj_err_threshold = 1e6;

// We're going to copy some tests from the projection factor test suite
// and keep them the same except operate on a cloned copy of the original factor
// graph

TEST(FactorGraphCloneTests, testUnfrozenCamerasWithCalibration)
{
    Eigen::Vector3d pt(10, 0, 0);
    Vector3dNodePtr landmark_node =
      util::allocate_aligned<Vector3dNode>(sym::L(0));

    FactorGraph graph;
    graph.addNode(landmark_node);

    Eigen::Quaterniond I_q_C(0.5, -0.5, 0.5, -0.5);
    Eigen::Vector3d I_p_C(0, 0.03, 0);
    Pose3 I_T_C(I_q_C, I_p_C);

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 1.0 / 10);

    boost::shared_ptr<CameraCalibration> calib =
      util::allocate_aligned<CameraCalibration>(
        500, 500, 0.0, 250, 250, 0, 0, 0, 0);

    MultiProjectionFactor::Ptr factor =
      util::allocate_aligned<MultiProjectionFactor>(
        landmark_node, I_T_C, calib, reproj_err_threshold);

    for (int i = 0; i < 5; ++i) {
        Eigen::Vector3d p(0, i - 2, 0);

        Pose3 pose(math::identity_quaternion(), p);

        SE3NodePtr node = util::allocate_aligned<SE3Node>(sym::X(i));
        node->pose() = Pose3(math::identity_quaternion(), p);

        Camera cam(pose.compose(I_T_C), calib);

        Eigen::Vector2d msmt = cam.project(pt);
        // msmt += Eigen::Vector2d(dist(generator), dist(generator));
        // std::cout << "msmt = " << msmt.transpose() << std::endl;

        Eigen::Matrix2d noise_cov = Eigen::Matrix2d::Identity();
        factor->addMeasurement(node, msmt, noise_cov);

        graph.addNode(node);

        CeresSE3PriorFactor::Ptr prior_fac =
          util::allocate_aligned<CeresSE3PriorFactor>(
            node, node->pose(), 0.1 * Eigen::MatrixXd::Identity(6, 6));
        graph.addFactor(prior_fac);
    }

    graph.addFactor(factor);

    landmark_node->vector() << 1, 0, 0;

    auto cloned_graph = graph.clone();

    cloned_graph->solve();

    auto cloned_landmark_node = cloned_graph->getNode<Vector3dNode>(sym::L(0));

    EXPECT_NEAR(landmark_node->vector()(0), 1, 1e-6);
    EXPECT_NEAR(landmark_node->vector()(1), 0, 1e-6);
    EXPECT_NEAR(landmark_node->vector()(2), 0, 1e-6);

    EXPECT_NEAR(cloned_landmark_node->vector()(0), 10, 1);
    EXPECT_NEAR(cloned_landmark_node->vector()(1), 0, 1);
    EXPECT_NEAR(cloned_landmark_node->vector()(2), 0, 1);

    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(0), 0, 1e-1);
        EXPECT_NEAR(cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(1),
                    i - 2,
                    1e-1);
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(2), 0, 1e-1);
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->rotation().w(), 1, 1e-1);
    }

    // auto cam1 = graph.getNode<SE3Node>(sym::X(0));

    // std::cout << cam1->pose() << std::endl;
}

TEST(FactorGraphCloneTests, testMultiFactor_MultiplePoints)
{
    aligned_vector<Eigen::Vector3d> pts{
        Eigen::Vector3d(10, 0, 0),  Eigen::Vector3d(9, -1, 2),
        Eigen::Vector3d(11, 2, -1), Eigen::Vector3d(10, -1, 1),
        Eigen::Vector3d(7, 3, 4),   Eigen::Vector3d(6, -3, -4),
        Eigen::Vector3d(12, 3, -4), Eigen::Vector3d(10, -3, 4)
    };

    FactorGraph graph;

    std::vector<Vector3dNodePtr> landmark_nodes;
    for (int i = 0; i < pts.size(); ++i) {
        landmark_nodes.push_back(
          util::allocate_aligned<Vector3dNode>(sym::L(i)));
        landmark_nodes[i]->vector() << 5, 0, 0;
        graph.addNode(landmark_nodes[i]);
    }

    Eigen::Quaterniond I_q_C(0.5, -0.5, 0.5, -0.5);
    Eigen::Vector3d I_p_C(0, 0.03, 0);
    Pose3 I_T_C(I_q_C, I_p_C);

    boost::shared_ptr<CameraCalibration> calib =
      util::allocate_aligned<CameraCalibration>(
        500, 500, 0.0, 250, 250, 0, 0, 0, 0);

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 1);

    std::vector<MultiProjectionFactor::Ptr> factors;

    for (int i = 0; i < pts.size(); ++i) {
        factors.push_back(util::allocate_aligned<MultiProjectionFactor>(
          landmark_nodes[i], I_T_C, calib, reproj_err_threshold));
    }

    for (int i = 0; i < 5; ++i) {
        Eigen::Vector3d p(0, i - 2, 0);

        Pose3 pose(math::identity_quaternion(), p);

        SE3NodePtr node = util::allocate_aligned<SE3Node>(sym::X(i));

        // node->pose() = Pose3(math::identity_quaternion(), p +
        // Eigen::Vector3d(0.2, 0.2, 0.2));

        graph.addNode(node);

        Camera cam(pose.compose(I_T_C), calib);
        Eigen::Matrix2d noise_cov = 1 * Eigen::Matrix2d::Identity();

        for (int j = 0; j < pts.size(); ++j) {
            Eigen::Vector2d msmt =
              cam.project(pts[j]) +
              Eigen::Vector2d(dist(generator), dist(generator));
            // std::cout << msmt.transpose() << std::endl;
            factors[j]->addMeasurement(node, msmt, noise_cov);
        }

        // Freeze first 2 nodes at true pose
        if (i == 0 || i == 1) {
            node->pose() = pose;
            graph.setNodeConstant(node);
        }

        // CeresSE3PriorFactor::Ptr prior_fac =
        //     util::allocate_aligned<CeresSE3PriorFactor>(node, node->pose(),
        //     0.1 * Eigen::MatrixXd::Identity(6,6));
        // graph.addFactor(prior_fac);
    }

    for (auto& f : factors) {
        graph.addFactor(f);
    }

    auto cloned_graph = graph.clone();

    cloned_graph->solve();

    for (int i = 0; i < 5; ++i) {
        // Expect two things -- that the cloned graph solved to the right
        // values, and that the original graph remains unmodified.

        // cloned correct
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(0), 0, 1e-1);
        EXPECT_NEAR(cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(1),
                    i - 2,
                    1e-1);
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(2), 0, 1e-1);
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->rotation().w(), 1, 1e-1);

        // original unmodified
        if (i > 2) {
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->translation()(0), 0, 1e-1);
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->translation()(1), 0, 1e-1);
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->translation()(2), 0, 1e-1);
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->rotation().w(), 1, 1e-1);
        }
    }

    auto cam1 = graph.getNode<SE3Node>(sym::X(4));

    // std::cout << cam1->pose() << std::endl;
}

TEST(FactorGraphCloneTests, testSmartFactor_MultiplePoints)
{
    aligned_vector<Eigen::Vector3d> pts{
        Eigen::Vector3d(10, 0, 0),  Eigen::Vector3d(9, -1, 2),
        Eigen::Vector3d(11, 2, -1), Eigen::Vector3d(10, -1, 1),
        Eigen::Vector3d(7, 3, 4),   Eigen::Vector3d(6, -3, -4),
        Eigen::Vector3d(12, 3, -4), Eigen::Vector3d(10, -3, 4)
    };
    // Eigen::Vector3d pt1(10, 0, 0);
    // Eigen::Vector3d pt2(9, -1, 2);
    // Eigen::Vector3d pt3(11, 2, -1);

    FactorGraph graph;

    Eigen::Quaterniond I_q_C(0.5, -0.5, 0.5, -0.5);
    Eigen::Vector3d I_p_C(0, 0.03, 0);
    Pose3 I_T_C(I_q_C, I_p_C);

    boost::shared_ptr<CameraCalibration> calib =
      util::allocate_aligned<CameraCalibration>(
        500, 500, 0.0, 250, 250, 0, 0, 0, 0);

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 0.5);

    std::vector<SmartProjectionFactor::Ptr> factors;

    for (auto& p : pts) {
        factors.push_back(util::allocate_aligned<SmartProjectionFactor>(
          I_T_C, calib, reproj_err_threshold));
    }

    for (int i = 0; i < 5; ++i) {
        Eigen::Vector3d p(0, i - 2, 0);

        Pose3 pose(math::identity_quaternion(), p);

        SE3NodePtr node = util::allocate_aligned<SE3Node>(sym::X(i));
        // node->pose() = Pose3(math::identity_quaternion(), p);
        graph.addNode(node);

        Camera cam(pose.compose(I_T_C), calib);
        Eigen::Matrix2d noise_cov = 0.25 * Eigen::Matrix2d::Identity();

        for (int j = 0; j < pts.size(); ++j) {
            Eigen::Vector2d msmt =
              cam.project(pts[j]) +
              Eigen::Vector2d(dist(generator), dist(generator));
            factors[j]->addMeasurement(node, msmt, noise_cov);
        }

        // Freeze first nodes
        if (i == 0 || i == 1) {
            node->pose() = pose;
            graph.setNodeConstant(node);
        }

        // CeresSE3PriorFactor::Ptr prior_fac =
        //     util::allocate_aligned<CeresSE3PriorFactor>(node, node->pose(),
        //     0.1 * Eigen::MatrixXd::Identity(6,6));
        // graph.addFactor(prior_fac);
    }

    for (auto& f : factors) {
        graph.addFactor(f);
    }

    auto cloned_graph = graph.clone();

    cloned_graph->solve();

    for (int i = 0; i < 5; ++i) {
        // check that cloned is correct
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(0), 0, 1e-1);
        EXPECT_NEAR(cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(1),
                    i - 2,
                    1e-1);
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->translation()(2), 0, 1e-1);
        EXPECT_NEAR(
          cloned_graph->getNode<SE3Node>(sym::X(i))->rotation().w(), 1, 1e-1);

        // check that original is unmodified
        if (i >= 2) {
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->translation()(0), 0, 1e-1);
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->translation()(1), 0, 1e-1);
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->translation()(2), 0, 1e-1);
            EXPECT_NEAR(
              graph.getNode<SE3Node>(sym::X(i))->rotation().w(), 1, 1e-1);
        }
    }

    // auto cam1 = graph.getNode<SE3Node>(sym::X(4));

    // std::cout << cam1->pose() << std::endl;
}

// Run all the tests that were declared with TEST()
int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}