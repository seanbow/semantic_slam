#include "semantic_slam/FactorGraph.h"

#include <gtsam/slam/PriorFactor.h>

#include <gtest/gtest.h>

namespace sym = gtsam::symbol_shorthand;

TEST(FactorGraphTests, testAddNode_CheckExistsTrue)
{
    FactorGraph graph;
    gtsam::Symbol symb = sym::C(3);
    NodeInfo node(symb);

    double value = 0;

    graph.addNode(node, value);

    auto result = graph.findNodeBySymbol(symb);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(result->symbol(), symb);
}

TEST(FactorGraphTests, testAddNode_CheckExistsFalse)
{
    FactorGraph graph;
    gtsam::Symbol symb = sym::C(3);
    NodeInfo node(symb);

    double value = 0;

    graph.addNode(node, value);

    auto result = graph.findNodeBySymbol(sym::C(4));
    
    EXPECT_FALSE(result);
}

TEST(FactorGraphTests, testFindNodeBeforeTime_Exists)
{
    FactorGraph graph;
    std::vector<NodeInfo> nodes;

    for (int i = 0; i <= 5; ++i) nodes.emplace_back(sym::C(i), ros::Time(i));

    double value = 0;

    for (int i = 0; i <= 5; ++i) graph.addNode(nodes[i], value);

    auto result = graph.findLastNodeBeforeTime('c', ros::Time(4));
    
    EXPECT_TRUE(result);
    EXPECT_EQ(result->symbol(), sym::C(4));
}

TEST(FactorGraphTests, testFindNodeBeforeTime_NotExists)
{
    FactorGraph graph;
    std::vector<NodeInfo> nodes;

    for (int i = 2; i <= 5; ++i) nodes.emplace_back(sym::C(i), ros::Time(i));

    double value = 0;

    for (int i = 2; i <= 5; ++i) graph.addNode(nodes[i], value);
    
    // wrong character
    auto result1 = graph.findLastNodeBeforeTime('b', ros::Time(3));
    EXPECT_FALSE(result1);

    // none in graph before time
    auto result2 = graph.findLastNodeBeforeTime('c', ros::Time(1));
    EXPECT_FALSE(result2);
}

TEST(FactorGraphTests, testSolve_SimplePriorFactor)
{
  FactorGraph graph;
  gtsam::Symbol symb = sym::X(0);
  double value = -1;

  NodeInfo node(symb);

  graph.addNode(node, value);

  auto noise_model = gtsam::noiseModel::Isotropic::Sigma(1,1);

  auto fac = util::allocate_aligned<gtsam::PriorFactor<double>>(symb, 0, noise_model);
  FactorInfo fac_info(FactorType::PRIOR, fac);

  graph.addFactor(fac_info);

  bool opt_succeeded = graph.solve();

  double result;
  bool get_result_succeeded = graph.getEstimate(symb, result);

  EXPECT_TRUE(opt_succeeded);
  EXPECT_TRUE(get_result_succeeded);
  EXPECT_NEAR(result, 0.0, 1e-8);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}