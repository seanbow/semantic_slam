#include "semantic_slam/FactorGraph.h"

#include <gtsam/slam/PriorFactor.h>

#include <gtest/gtest.h>

namespace sym = gtsam::symbol_shorthand;

class FactorGraphTests : public ::testing::Test
{
protected:
  void SetUp() {
    std::vector<NodeInfoPtr> nodes;
    double value = 0;

    for (int i = 2; i <= 5; ++i) {
      nodes.emplace_back(NodeInfo::Create(sym::C(i), ros::Time(i)));
      graph.addNode(nodes.back(), value);
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
    auto result = graph.findNodeBySymbol(sym::C(3));
    
    EXPECT_TRUE(result);
    EXPECT_EQ(result->symbol(), sym::C(3));
}

TEST_F(FactorGraphTests, testAddNode_CheckExistsFalse)
{
    auto result = graph.findNodeBySymbol(sym::C(6));
    
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
  gtsam::Symbol symb = sym::X(0);
  double value = -1;

  NodeInfoPtr node = NodeInfo::Create(symb);

  graph2.addNode(node, value);

  auto noise_model = gtsam::noiseModel::Isotropic::Sigma(1,1);

  auto fac = util::allocate_aligned<gtsam::PriorFactor<double>>(symb, 0, noise_model);
  FactorInfoPtr fac_info = FactorInfo::Create(FactorType::PRIOR, fac);

  graph2.addFactor(fac_info);

  bool opt_succeeded = graph2.solve();

  double result;
  bool get_result_succeeded = graph2.getEstimate(symb, result);

  EXPECT_TRUE(opt_succeeded);
  EXPECT_TRUE(get_result_succeeded);
  EXPECT_NEAR(result, 0.0, 1e-8);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}