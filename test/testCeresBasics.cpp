#include "semantic_slam/SE3Node.h"
#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>

#include <gtest/gtest.h>

#include "semantic_slam/CeresFactorGraph.h"
#include "semantic_slam/CeresSE3PriorFactor.h"
#include "semantic_slam/CeresBetweenFactor.h"

#include "test_utils.h"

namespace sym = gtsam::symbol_shorthand;

// class FactorGraphTests : public ::testing::Test
// {
// protected:
//   void SetUp() {
//     std::vector<NodeInfoPtr> nodes;
//     double value = 0;

//     for (int i = 2; i <= 5; ++i) {
//       nodes.emplace_back(NodeInfo::Create(sym::C(i), ros::Time(i)));
//       graph.addNode(nodes.back(), value);
//     }

//   }

//   FactorGraph graph;
// };

/******************************/

// TEST_F(FactorGraphTests, testNumNodes_CheckEqual)
// {
//   EXPECT_TRUE(graph.num_nodes() == 4);
// }

TEST(CeresBasicTests, testAddSE3Node_Exists)
{
    boost::shared_ptr<ceres::Problem> problem = boost::make_shared<ceres::Problem>();
    SE3Node node(gtsam::Symbol('x', 0));

    node.addToProblem(problem);

    EXPECT_TRUE(problem->NumParameters() == 7);
    EXPECT_TRUE(problem->NumResiduals() == 0);
    EXPECT_TRUE(problem->HasParameterBlock(node.pose().rotation_data()));
}

TEST(CeresBasicTests, testAddNodeToGraph_CheckKeys)
{
    CeresFactorGraph graph;

    SE3NodePtr node0 = boost::make_shared<SE3Node>(sym::X(0));
    SE3NodePtr node1 = boost::make_shared<SE3Node>(sym::X(1));

    graph.addNode(node0);
    graph.addNode(node1);

    std::vector<gtsam::Key> keys = graph.keys();

    const ceres::Problem& problem = graph.problem();

    EXPECT_TRUE(problem.NumParameters() == 14);
    EXPECT_TRUE(problem.HasParameterBlock(node1->pose().rotation_data()));
    EXPECT_TRUE(std::find(keys.begin(), keys.end(), node0->key()) != keys.end());
}

TEST(CeresBasicTests, testAddFactor_checkExistsInProblem)
{
    CeresFactorGraph graph;
    SE3NodePtr node0 = boost::make_shared<SE3Node>(sym::X(0));

    Pose3 prior = Pose3::Identity();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6,6);
    CeresSE3PriorFactorPtr factor = boost::make_shared<CeresSE3PriorFactor>(node0, prior, cov);

    graph.addNode(node0);
    graph.addFactor(factor);

    EXPECT_TRUE(graph.problem().NumResiduals() == 6);
}

TEST(CeresBasicTests, testSolve_SimplePriorProblem)
{
    CeresFactorGraph graph;
    SE3NodePtr node0 = boost::make_shared<SE3Node>(sym::X(0));

    Pose3 prior = Pose3::Identity();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6,6);
    CeresSE3PriorFactorPtr factor = boost::make_shared<CeresSE3PriorFactor>(node0, prior, cov);

    graph.addNode(node0);
    graph.addFactor(factor);

    node0->pose().translation() << 1, 0, 0;

    EXPECT_NEAR(node0->pose().translation().norm(), 1.0, 1e-8);

    graph.solve();

    EXPECT_NEAR(node0->pose().translation().norm(), 0.0, 1e-8);
}

// Sanity check on Jacobian calculations
// Check that a "between" Jacobian is the same as inverse and multiply
TEST(CeresBasicTests, testJacobians_BetweenInverseEquivalence)
{
    Pose3 x0;
    x0.translation() << 1, -4.2, 3;
    x0.rotation() << 0, .2, .45, .4;
    x0.rotation().normalize();

    Pose3 x1;
    x1.translation() << -4.7, 3.2, 1.04;
    x1.rotation() << -.1, .76, -.2, .17;
    x1.rotation().normalize();

    // Compute Jacobian with between function
    Eigen::MatrixXd H0, H1;
    Pose3 dx = x0.between(x1, H0, H1);

    // Compute Jacobian by inversion & multiplication
    Eigen::MatrixXd Hinv0;
    Eigen::MatrixXd Hmul0, Hmul1;
    Pose3 x0_inv = x0.inverse(Hinv0);
    Pose3 dx_mul = x0_inv.compose(x1, Hmul0, Hmul1);

    Eigen::MatrixXd H_final_0 = Hmul0 * Hinv0;
    Eigen::MatrixXd H_final_1 = Hmul1;

    EXPECT_TRUE(H_final_0.isApprox(H0));
    EXPECT_TRUE(H_final_1.isApprox(H1));
}

// Simple between factor test case
TEST(CeresBasicTests, testBetweenFactor_SimpleCase)
{
    // Initialize poses at origin, say true pose of 
    // x1 is at (1,0,0) but only enforce with between factor
    SE3NodePtr x0 = boost::make_shared<SE3Node>(sym::X(0));
    SE3NodePtr x1 = boost::make_shared<SE3Node>(sym::X(1));

    x0->pose() = Pose3::Identity();
    x1->pose() = Pose3::Identity();

    // Anchor x0 at origin
    Eigen::MatrixXd prior_cov = 1e-4 * Eigen::MatrixXd::Identity(6,6);
    CeresSE3PriorFactorPtr x0_prior = boost::make_shared<CeresSE3PriorFactor>(x0, Pose3::Identity(), prior_cov);

    CeresFactorGraph graph;

    graph.addNode(x0);
    graph.addNode(x1);
    graph.addFactor(x0_prior);

    // Between factor between x0 and x1
    Eigen::MatrixXd between_cov = 1e-1 * Eigen::MatrixXd::Identity(6,6);
    Pose3 between;
    between.rotation() << 0, 0, 0, 1;
    between.translation() << 1, 1, 1;

    CeresBetweenFactorPtr fac = boost::make_shared<CeresBetweenFactor>(x0,x1,between,between_cov);

    graph.addFactor(fac);

    graph.solve();

    EXPECT_TRUE(x1->translation().isApprox(between.translation(), 1e-8));
}

TEST(CeresBasicTests, testBetweenCostTerm_Result)
{
    math::Quaternion q1, q2;
    Eigen::Vector3d p1, p2;

    q1 << -.1, 0.6, -0.58, 0.34;
    p1 << 1, -3.5, 0.25;

    // q2 << 0, 0, 0, 1;
    q2 << 0.4, 0.75, -0.31, -0.15;
    p2 << -4.8, -0.5, -7;

    q1.normalize();
    q2.normalize();

    Eigen::Quaterniond q1e = Eigen::Map<Eigen::Quaterniond>(q1.data());
    Eigen::Quaterniond q2e = Eigen::Map<Eigen::Quaterniond>(q2.data());

    Eigen::VectorXd residual(6);

    Pose3 identity_pose = Pose3::Identity();

    ceres::CostFunction* cost = BetweenCostTerm::Create(identity_pose, 1 * Eigen::MatrixXd::Identity(6,6));

    double *parameters[] = {q1.data(),
                    p1.data(),
                    q2.data(),
                    p2.data()};

    std::vector<const ceres::LocalParameterization*> parameterizations;
    parameterizations.push_back(new QuaternionLocalParameterization);
    parameterizations.push_back(NULL);
    parameterizations.push_back(new QuaternionLocalParameterization);
    parameterizations.push_back(NULL);

    ceres::NumericDiffOptions opts;

    ceres::GradientChecker checker(cost, &parameterizations, opts);

    ceres::GradientChecker::ProbeResults results;

    EXPECT_TRUE(checker.Probe(parameters, 1e-3, &results));
    
    std::cout << "CHECKER: " << results.error_log << std::endl;

    math::Quaternion q12 = math::quat_mul(math::quat_inv(q1), q2);
    Eigen::Quaterniond q12e = q1e.conjugate() * q2e;

    std::cout << "q12:\n" << q12.transpose()  << std::endl;
    std::cout << "EIGEN q12 conj:\n " << q12e.vec().transpose() << " " << q12e.w() << std::endl;

    std::cout << "qidentity*qa2inv:\n" << math::quat_mul(math::identity_quaternion(),math::quat_inv(q12)) << std::endl;

    Eigen::Vector3d x;
    x << 1.5, -2.5, 6;

    std::cout << "EIGEN" << std::endl;
    std::cout << "(q1*q2)*x = " << ((q1e*q2e)*x).transpose() << std::endl;
    std::cout << "q1 * (q2*x) = " << (q1e * (q2e*x)).transpose() << std::endl;

    std::cout << "EIGEN inv:" << std::endl;
    std::cout << "(q1*q2)*x = " << ((q1e.conjugate()*q2e.conjugate())*x).transpose() << std::endl;
    std::cout << "q1 * (q2*x) = " << (q1e.conjugate() * (q2e.conjugate()*x)).transpose() << std::endl;

    std::cout << "ME" << std::endl;
    std::cout << "(q1*q2)*x = " << (math::quat2rot(math::quat_mul(q1,q2))*x).transpose() << std::endl;
    std::cout << "q1 * (q2*x) = " << (math::quat2rot(q1) * (math::quat2rot(q2)*x)).transpose() << std::endl;

    // cost.Evaluate(q1.data(),
    //                 p1.data(),
    //                 q2.data(),
    //                 p2.data(),
    //                 residual.data(), NULL);

    // cost.Evaluate(parameters, residual.data(), NULL);

    // std::cout << "between residual:\n" << residual << std::endl;

    // compare to gtsam
    // Eigen::Map<Eigen::Quaterniond> q1_e(q1.data());
    // Eigen::Map<Eigen::Quaterniond> q2_e(q2.data());

    // gtsam::Pose3 x1(gtsam::Rot3(q1_e), p1);
    // gtsam::Pose3 x2(gtsam::Rot3(q2_e), p2);

    // // std::cout << "GTSAM pose1: " << x1 << "\n pose2:\n" << x2 << std::endl;

    // gtsam::Pose3 between_gt = x1.between(x2);

    // Pose3 x1_ours(q1,p1);
    // Pose3 x2_ours(q2,p2);
    // Pose3 between_ours = x1_ours.between(x2_ours);

    // EXPECT_TRUE(residual.head<3>().isApprox(2.0 * between_gt.rotation().toQuaternion().vec(), 1e-4));

    // std::cout << "gtsam between: \n" << between_gt.rotation().toQuaternion().vec() << "\n" << between_gt.translation() << std::endl;

    // std::cout << "Ours between:\n" << between_ours.rotation() << "\n" << between_ours.translation() << std::endl;

    EXPECT_TRUE(true);
}

// Simple between factor test case
TEST(CeresBasicTests, testBetweenFactor_SimpleCase2)
{
    SE3NodePtr x0 = boost::make_shared<SE3Node>(sym::X(0));
    SE3NodePtr x1 = boost::make_shared<SE3Node>(sym::X(1));

    x0->pose().translation() << 1, -3.5, 0.25;
    x0->pose().rotation() << -.1, 0.69, -0.58, 0.34;
    // x0->pose().rotation() << 0, 0, 0, 1;
    x0->pose().rotation().normalize();

    x1->pose() = Pose3::Identity();

    // Anchor x0
    Eigen::MatrixXd prior_cov = 1e-4 * Eigen::MatrixXd::Identity(6,6);
    Pose3 x0_prior = x0->pose();
    CeresSE3PriorFactorPtr x0_prior_fac = boost::make_shared<CeresSE3PriorFactor>(x0, x0_prior, prior_cov);

    CeresFactorGraph graph;

    graph.addNode(x0);
    graph.addNode(x1);
    graph.addFactor(x0_prior_fac);

    // Between factor between x0 and x1
    Eigen::MatrixXd between_cov = 1e-2 * Eigen::MatrixXd::Identity(6,6);
    Pose3 between;
    between.rotation() << 0.4, 0.75, -0.31, 0.15;
    between.rotation().normalize();
    between.translation() << -4.8, -0.35, -7;

    CeresBetweenFactorPtr fac = boost::make_shared<CeresBetweenFactor>(x0,x1,between,between_cov);

    graph.addFactor(fac);

    graph.solve();

    // Compute "truth" answer with simple multiplication
    Pose3 truth_x1 = x0_prior.compose(between);

    std::cout << "X1 trans:\n" << x1->translation() << std::endl;
    std::cout << "X1 truth trans: \n" << truth_x1.translation() << std::endl;

    std::cout << "X1 rot:\n" << x1->rotation() << std::endl;
    std::cout << "X1 truth rot: \n" << truth_x1.rotation() << std::endl;

    EXPECT_TRUE(x1->translation().isApprox(truth_x1.translation(), 1e-3));
    EXPECT_TRUE(x1->rotation().isApprox(truth_x1.rotation(), 1e-3));
}

TEST(CeresBasicTests, testBetweenSolve_g2o_Sphere2500)
{
    std::unordered_map<gtsam::Key, SE3NodePtr> nodes;
    std::vector<CeresFactorPtr> factors;
    std::string filename("/home/sean/code/SESync/data/sphere2500.g2o");

    readG2oFile(filename, nodes, factors);

    std::cout << "g2o file has " << nodes.size() << " nodes and " << factors.size() << " factors." << std::endl;

    outputPoses("initial_poses.txt", nodes);

    CeresFactorGraph graph;

    for (auto& node_pair : nodes) {
        graph.addNode(node_pair.second);
    }

    for (auto& factor : factors) {
        graph.addFactor(factor);
    }

    graph.setFirstNodeConstant();

    graph.solve(true);

    outputPoses("optimized_poses.txt", nodes);

    EXPECT_TRUE(true);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}