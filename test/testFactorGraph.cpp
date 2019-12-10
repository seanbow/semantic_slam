#include "semantic_slam/FactorGraph.h"

// #include <gtsam/slam/PriorFactor.h>
#include "semantic_slam/CeresSE3PriorFactor.h"
#include "semantic_slam/CeresVectorPriorFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/VectorNode.h"
#include <gtest/gtest.h>

namespace sym = symbol_shorthand;

using NodeType = VectorNode<2>;

class FactorGraphTests : public ::testing::Test
{
  protected:
    void SetUp()
    {
        std::vector<CeresNodePtr> nodes;
        double value = 0;

        for (int i = 2; i <= 5; ++i) {
            nodes.emplace_back(
              util::allocate_aligned<NodeType>(sym::C(i), ros::Time(i)));
            graph.addNode(nodes.back());
        }
    }

    FactorGraph graph;
};

/******************************/

TEST_F(FactorGraphTests, testNumNodes_CheckEqual)
{
    EXPECT_TRUE(graph.num_nodes() == 4);
}

TEST_F(FactorGraphTests, testAddNode_CheckExistsTrue)
{
    auto result = graph.getNode(sym::C(3));

    EXPECT_TRUE(result);
    EXPECT_EQ(result->symbol(), sym::C(3));
}

TEST_F(FactorGraphTests, testAddNode_CheckExistsFalse)
{
    auto result = graph.getNode(sym::C(6));

    EXPECT_FALSE(result);
}

/******************************/

TEST_F(FactorGraphTests, testFindNodeBeforeTime_Exists)
{
    auto result = graph.findLastNodeBeforeTime('c', ros::Time(4.25));

    EXPECT_TRUE(result);
    EXPECT_EQ(result->symbol(), sym::C(4));
}

TEST_F(FactorGraphTests, testFindNodeBeforeTime_NotExists)
{
    // wrong character
    auto result1 = graph.findLastNodeBeforeTime('b', ros::Time(3));
    EXPECT_FALSE(result1);

    // none in graph before time
    auto result2 = graph.findLastNodeBeforeTime('c', ros::Time(1));
    EXPECT_FALSE(result2);
}

/******************************/

TEST_F(FactorGraphTests, testFindNodeAfterTime_Exists)
{
    auto result = graph.findFirstNodeAfterTime('c', ros::Time(3.5));

    EXPECT_TRUE(result);
    EXPECT_EQ(result->symbol(), sym::C(4));
}

TEST_F(FactorGraphTests, testFindNodeAfterTime_NotExists)
{
    // wrong character
    auto result1 = graph.findFirstNodeAfterTime('b', ros::Time(3));
    EXPECT_FALSE(result1);

    // none in graph after time
    auto result2 = graph.findFirstNodeAfterTime('c', ros::Time(5.5));
    EXPECT_FALSE(result2);
}

/******************************/

TEST_F(FactorGraphTests, testFindLastNode_Exists)
{
    auto result = graph.findLastNode('c');

    EXPECT_TRUE(result);
    EXPECT_EQ(result->symbol(), sym::C(5));
}

TEST_F(FactorGraphTests, testFindLastNode_NotExists)
{
    auto result = graph.findLastNode('d');

    EXPECT_FALSE(result);
}

TEST(FactorGraphTest, testFindLastNode_EmptyGraph)
{
    FactorGraph graph2;

    auto result = graph2.findLastNode('c');

    EXPECT_FALSE(result);
}

/******************************/

TEST(FactorGraphTest, testSolve_SimplePriorFactor)
{
    FactorGraph graph2;
    Symbol symb = sym::X(0);

    Vector2dNodePtr node = util::allocate_aligned<VectorNode<2>>(symb);
    node->vector() << 0, 0;

    graph2.addNode(node);

    Eigen::Vector2d prior;
    prior << 2, -4;
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();

    auto fac =
      util::allocate_aligned<CeresVector2dPriorFactor>(node, prior, cov);

    graph2.addFactor(fac);

    graph2.solve(false);

    EXPECT_TRUE(node->vector().isApprox(prior, 1e-8));
}

/******************************/

TEST(FactorGraphTest, testClone_SimplePriorExample)
{
    FactorGraph graph2;
    Symbol symb = sym::X(0);

    Vector2dNodePtr node = util::allocate_aligned<VectorNode<2>>(symb);
    node->vector() << 0, 0;

    graph2.addNode(node);

    Eigen::Vector2d prior;
    prior << 2, -4;
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();

    auto fac =
      util::allocate_aligned<CeresVector2dPriorFactor>(node, prior, cov);

    graph2.addFactor(fac);

    boost::shared_ptr<FactorGraph> cloned_graph = graph2.clone();

    graph2.solve(false);

    auto new_node = cloned_graph->getNode<Vector2dNode>(sym::X(0));

    // Check that the new node remains unsolved...
    EXPECT_TRUE(node != new_node);
    EXPECT_TRUE(new_node->vector().norm() < 1e-10);
    EXPECT_TRUE(node->vector().isApprox(prior, 1e-8));

    // Check solving the cloned graph
    cloned_graph->solve(false);
    EXPECT_TRUE(new_node->vector().isApprox(prior, 1e-8));
}

/******************************/

TEST(FactorGraphTest, testSolve_SE3PriorFactor)
{
    FactorGraph graph2;
    Symbol symb = sym::X(0);

    SE3NodePtr node = util::allocate_aligned<SE3Node>(symb);

    graph2.addNode(node);

    Pose3 prior;
    prior.rotation() = Eigen::Quaterniond(.5, -.5, .5, -.5);
    prior.rotation().normalize();
    prior.translation() << 1, 1, 1;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);

    auto fac = util::allocate_aligned<CeresSE3PriorFactor>(node, prior, cov);

    graph2.addFactor(fac);

    graph2.solve(false);

    EXPECT_TRUE(node->pose().translation().isApprox(prior.translation(), 1e-8));
    EXPECT_TRUE(node->pose().rotation().coeffs().isApprox(
      prior.rotation().coeffs(), 1e-8));
}

// Run all the tests that were declared with TEST()
int
main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}