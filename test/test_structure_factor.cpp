
#include <eigen3/Eigen/Dense>

#include "semantic_slam/keypoints/gtsam/StructureFactor.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>

#include <iostream>

using std::cout;
using std::endl;

// #include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

namespace sym = gtsam::symbol_shorthand;

class StructureFactorTester
{
  public:
    void run()
    {
        // test data

        size_t m = 12;
        size_t k = 2;

        Eigen::MatrixXd P(3, 12);
        P << 19.6106000000000016, 18.9085000000000001, 17.2723000000000013,
          0.0000000000000000, 18.5507999999999988, 13.5353999999999992,
          16.1299999999999990, 18.2070000000000007, 20.1805999999999983,
          18.2868999999999993, 0.0000000000000000, 16.2493000000000016,
          3.1941700000000002, 4.6080199999999998, 2.1755399999999998,
          0.0000000000000000, 3.5149599999999999, 2.0985900000000002,
          3.4544899999999998, 4.2941300000000000, 2.7403800000000000,
          2.0763600000000002, 0.0000000000000000, 3.9458299999999999,
          -0.2617770000000000, -0.1951060000000000, -0.1099920000000000,
          0.0000000000000000, 0.3452420000000000, 0.2706040000000000,
          0.3623350000000000, 0.4083230000000000, -0.0699446000000000,
          -0.0100630000000000, 0.0000000000000000, 0.1543300000000000;

        geometry::ObjectModelBasis model;

        model.mu = Eigen::MatrixXd(3, 12);
        model.mu << -0.9338585488534954, -0.9440181466519368,
          0.9335527065805910, 0.9437451054615971, -0.6201054991851856,
          0.6171470212234791, 0.5861163860230332, -0.5838797335651555,
          -0.6855697941610017, 0.6866993712843926, 0.7270985501883458,
          -0.7269274183446639, -1.7500719672045963, 1.2937682175480423,
          -1.7498000873731594, 1.2948877238682706, -0.6105020527446646,
          -0.6052512380958747, 1.3437791135098487, 1.3441703602716908,
          -2.4812107561583616, -2.4808807100073107, 2.2005556981930576,
          2.2005556981930576, -0.5743870637120234, -0.5721604481404006,
          -0.5748003144226467, -0.5743261686844970, 0.6065386596351594,
          0.6097537213137784, 0.6142850644964245, 0.6144861522175948,
          -0.1768231008778597, -0.1813437495660182, 0.1043886238702442,
          0.1043886238702442;

        model.pc = Eigen::MatrixXd(6, 12);
        model.pc << -0.0281725304679886, -0.0493742724179996,
          0.0267949346629413, 0.0468304151248244, -0.1103112257217705,
          0.0944098312375421, -0.0102220698281515, 0.0165388122072886,
          -0.0347123670644854, 0.0407720368229393, 0.0734189034167010,
          -0.0659724679718407, -0.2836077645784356, -0.1449466012075568,
          -0.2837184631250799, -0.1495951777375648, -0.5685506315422850,
          -0.5476578869706957, 1.4398253833434347, 1.4406597570866417,
          -0.2747986636959340, -0.2758530218724389, -0.1758784648500424,
          -0.1758784648500424, -0.1176107311522425, -0.1308298733083225,
          -0.1179988298778871, -0.1170583857667667, 0.1698245876511601,
          0.1832834677376640, 0.1695142449822566, 0.1697893033114740,
          0.0025787527634829, 0.0334531954718392, -0.1224728659063290,
          -0.1224728659063290, 0.0432287097760659, 0.0412402927352889,
          -0.0405210773731732, -0.0398718173988305, 0.0579060816206258,
          -0.0485841620093531, -0.1011264086503686, 0.0998498576675626,
          0.0144137844799573, -0.0160931407119248, -0.0909745204220315,
          0.0805324002861819, 0.0523779679136310, -0.3111636361606935,
          0.0525278720339883, -0.3156863218322823, 0.2651956212885036,
          0.2452274508734826, 0.1467111966195829, 0.1481662651158488,
          -0.1114478314503299, -0.1127460423110851, -0.0295812710453233,
          -0.0295812710453233, 0.2152594775953444, 0.2014399886024142,
          0.2145217360755592, 0.2159430936524297, -0.2156557797402336,
          -0.2267208611555844, -0.2079803504703635, -0.2079762133879910,
          0.0041760538713272, 0.0332159100089961, -0.0131115275259492,
          -0.0131115275259492;

        Eigen::VectorXd weights(12);
        Eigen::Matrix3d R0;
        Eigen::Vector3d t0;

        // weights << 0.2231674059983915, 0.0810473556263949,
        // 0.0690928250240069, 0.0000000000000000,
        //         0.1113884591936109, 0.0097218605294454, 0.0285835295423834,
        //         0.0578638555475560, 0.3185693917102257, 0.0488903123033908,
        //         0.0000000000000000, 0.0516750045245944;

        // R0 << -0.322327, -0.946585, -0.009087,
        //         -0.871256, 0.292896, 0.393859,
        //         -0.370160, 0.134868, -0.919126;
        // t0 << 14.744283,
        //         2.675206,
        //         0.074496;

        weights = Eigen::VectorXd::Ones(m);
        R0 = Eigen::Matrix3d::Identity();
        t0 << 0, 0, 0;

        gtsam::Pose3 pose0 = gtsam::Pose3(gtsam::Rot3(R0), gtsam::Point3(t0));

        std::vector<gtsam::Key> landmark_keys;

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values;

        for (size_t i = 0; i < m; ++i) {
            landmark_keys.push_back(sym::L(i));

            values.insert(sym::L(i), gtsam::Point3(P.col(i)));

            // skip priors where we have no measurements
            if (i != 3 && i != 10) {

                Eigen::Vector3d prior_noise = Eigen::Vector3d::Constant(0.5);

                gtsam::noiseModel::Diagonal::shared_ptr prior_model =
                  gtsam::noiseModel::Diagonal::Sigmas(prior_noise);

                gtsam::PriorFactor<gtsam::Point3> prior(
                  sym::L(i), gtsam::Point3(P.col(i)), prior_model);

                graph.push_back(prior);
            }
        }

        values.insert(sym::O(0), pose0);

        gtsam::Vector c0 = Eigen::VectorXd::Zero(k);

        values.insert(sym::C(0), c0);

        semslam::StructureFactor sf(
          sym::O(0), landmark_keys, sym::C(0), model, weights);

        cout << "Hobject: " << endl << sf.Dobject(values) << endl;

        graph.push_back(sf);

        // graph.print("\nFactor Graph:\n");
        // values.print("\nInitial Estimate:\n");

        // auto linearized = sf.linearize(values);

        // linearized->print("\nLinearized at x0:\n");

        // auto S = sf.structure(values);
        // cout << "S0 = \n" << S << endl  << endl;

        Eigen::VectorXd unwhitened = sf.unwhitenedError(values);
        Eigen::VectorXd residual = sf.whitenedError(values);

        Eigen::MatrixXd errors(residual.size(), 2);
        errors.leftCols(1) = unwhitened;
        errors.rightCols(1) = residual;

        cout << "unwhitened/whitened residual: \n" << errors << endl << endl;

        gtsam::LevenbergMarquardtParams lm_params;
        lm_params.setVerbosityLM("SUMMARY");
        // lm_params.setVerbosityLM("TRYLAMBDA");
        // lm_params.setVerbosity("LINEAR");
        lm_params.diagonalDamping = true;
        // lm_params.useFixedLambdaFactor = false;

        gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);

        // optimizer.iterate();
        // cout << " New values: \n";
        // optimizer.values().print();
        // cout << endl;
        // cout << "new residual: \n" << sf.whitenedError(optimizer.values()) <<
        // endl << endl; cout << "new structure: \n" <<
        // sf.structure(optimizer.values()) << endl << endl;

        // optimizer.iterate();
        // cout << " New values: \n";
        // optimizer.values().print();
        // cout << endl;
        // cout << "new residual: \n" << sf.whitenedError(optimizer.values()) <<
        // endl << endl;

        gtsam::Values result = optimizer.optimize();
        result.print("\nFinal result:\n");
    }
};

int
main()
{

    StructureFactorTester sft;

    sft.run();

    return 0;
}