#include "semantic_slam/Utils.h"
#include "semantic_slam/keypoints/geometry.h"

#include <iostream>

int
main()
{
    Eigen::MatrixXd A(6, 4);

    using std::cout;
    using std::endl;

    using namespace geometry;

    A << -0.734169112696739, 2.023690886603053, -0.590034564205221,
      0.066190048424611, -0.030813730012320, -2.258353970496191,
      -0.278064163765309, 0.652355888661374, 0.232347012624477,
      2.229445680456899, 0.422715691220478, 0.327059967177088,
      0.426387557408945, 0.337563700613106, -1.670200697850470,
      1.082633504236756, -0.372808741723504, 1.000060819589125,
      0.471634326416303, 1.006077110819051, -0.236454583757186,
      -1.664164474987060, -1.212847199674459, -0.650907736597753;

    Eigen::MatrixXd P(3, 12);

    P << 16.8125, 16.0169, 14.7361, 0, 16.7413, 9.5373, 15.6507, 15.4707,
      17.0730, 14.8902, 0, 15.2891, 2.6407, 3.7799, 1.7937, 0, 3.0905, 1.2496,
      3.3641, 3.5368, 2.2410, 1.5655, 0, 3.6842, -0.2028, -0.1304, -0.0821, 0,
      0.3001, 0.2040, 0.3629, 0.3542, -0.0546, -0.0030, 0, 0.1547;

    Eigen::VectorXd w(12);
    w << 0.1533, 0.0645, 0.1234, 0, 0.1081, 0.0463, 0.1144, 0.0538, 0.2066,
      0.0563, 0, 0.0733;

    Eigen::MatrixXd xyn(3, 12);
    xyn << -0.2289482601395690, -0.3222884120181385, -0.1907636525528814,
      -0.3081459647638098, -0.2614758888245250, -0.2473334415701963,
      -0.3152171883909742, -0.3152171883909742, -0.1978348761800458,
      -0.1794496947494184, -0.3109744542146756, -0.3251169014690043,
      0.0196432935383724, 0.0182290488129396, 0.0111578251857752,
      0.0125720699112081, -0.0157128245974494, -0.0142985798720165,
      -0.0171270693228823, -0.0171270693228823, 0.0083293357349094,
      0.0055008462840437, -0.0043988667939864, -0.0029846220685535,
      1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
      1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
      1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
      1.0000000000000000, 1.0000000000000000, 1.0000000000000000;

    xyn.array().rowwise() /= xyn.colwise().norm().array();

    Eigen::VectorXd scores(12);
    scores << 0.6266383528709412, 0.7167085409164429, 0.1346543282270432,
      0.0282017439603806, 0.6706644892692566, 0.0681838989257812,
      0.0756485462188721, 0.3222227692604065, 0.6471698284149170,
      0.3548924624919891, 0.0350466221570969, 0.2004920095205307;

    std::cout << "A: " << std::endl << A << std::endl;
    std::cout << "P: " << std::endl << P << std::endl;

    std::cout << "reshape A: " << std::endl << reshapeS_b2v(A) << std::endl;

    std::cout << "centralize A: " << std::endl << centralize(A) << std::endl;

    cout << "xyn: " << endl << xyn << endl;

    cout << "std(xyn,1,2) = " << endl << sample_stddev(xyn) << endl;

    std::string model_path(
      "/home/sean/code/old_semslam/semslam/models/car_basis.dat");

    ObjectModelBasis model = readModelFile(model_path);

    std::cout << "Model pc = " << std::endl << model.pc << std::endl;

    cout << endl << endl << endl;

    Eigen::Vector2d C(-.3726, .2402);

    cout << "C = " << C.transpose() << endl;

    cout << "composeShape(pc, C) = " << endl
         << composeShape(model.pc, C) << endl
         << endl;

    // TIME_TIC;
    StructureResult res;
    // res = optimizeStructure(P, model, w);
    // TIME_TOC("Levenberg-marquardt structure optimization from 3d points");

    // std::cout << "Optimization result: R = " << std::endl << res.R <<
    // std::endl; cout << "t = " << res.t.transpose() << endl; cout << " C = "
    // << res.C.transpose() << endl;

    // TIME_TIC;
    // res = optimizeStructureCoordinateDescent(P, model, w);
    // TIME_TOC("Coordinate descent structure optimization from 3d points");

    // std::cout << "Optimization result: R = " << std::endl << res.R <<
    // std::endl; cout << "t = " << res.t.transpose() << endl; cout << " C = "
    // << res.C.transpose() << endl;

    cout << "Optimizing from projections only: " << endl;
    TIME_TIC;
    res = optimizeStructureFromProjection(xyn, model, scores);
    TIME_TOC("Full projection structure optimization");

    std::cout << "Optimization result: R = " << std::endl << res.R << std::endl;
    cout << "t = " << res.t.transpose() << endl;
    cout << " C = " << res.C.transpose() << endl;

    cout << "Z = " << endl << res.Z.transpose() << endl;

    /**  ***  **/
    /** CERES **/
    /**  ***  **/
    // geometry_ceres::StructureResult resc;
    // resc = geometry_ceres::optimizeStructureFromProjection(xyn, model,
    // scores);

    // std::cout << "Optimization result: R = " << std::endl << resc.R <<
    // std::endl; cout << "t = " << resc.t.transpose() << endl; cout << " C = "
    // << resc.C.transpose() << endl;

    // cout << "Z = " << endl << resc.Z.transpose() << endl;

    /**              **             **/
    /** NULL PC AND KARL'S NEW DATA **/
    /**              **             **/

    std::string msg = "Running with null pc and Karl's data";

    cout << std::string(msg.length() + 6, '*') << endl;
    cout << "** " << msg << " **" << endl;
    cout << std::string(msg.length() + 6, '*') << endl << endl;

    model.mu = Eigen::MatrixXd::Zero(3, 10);
    model.mu << 0.0929521000000000, -0.0729283000000000, -0.0881835000000000,
      0.0858484000000000, 0.0579872000000000, -0.0466112000000000,
      -0.0937917000000000, 0.0726916000000000, -0.0081129300000000,
      -0.0003045900000000, -0.0993881000000000, -0.1114550000000000,
      0.1318720000000000, 0.1406520000000000, -0.1215500000000000,
      -0.1285600000000000, 0.1390610000000000, 0.1548260000000000,
      0.0744569000000000, -0.2322850000000000, -0.1506010000000000,
      -0.1533490000000000, -0.1539640000000000, -0.1495560000000000,
      0.0511484000000000, 0.0529216000000000, -0.0055370900000000,
      -0.0040530500000000, 0.1346830000000000, 0.2109200000000000;

    model.pc = Eigen::MatrixXd::Zero(0, 10);

    xyn = Eigen::MatrixXd::Zero(3, 10);
    xyn << -0.1095833300000000, -0.1376366600000000, 0.0096433330000000,
      0.0324366650000000, -0.1481566600000000, -0.1534166600000000,
      0.0219166650000000, 0.0166566670000000, -0.0271766650000000,
      -0.2410833200000000, 0.1779633300000000, 0.1306233300000000,
      0.1113366600000000, 0.1376366600000000, 0.0710100010000000,
      0.0569833330000000, 0.0429566650000000, 0.0429566650000000,
      -0.0131500000000000, -0.0078900000000000, 1.0000000000000000,
      1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
      1.0000000000000000, 1.0000000000000000, 1.0000000000000000,
      1.0000000000000000, 1.0000000000000000, 1.0000000000000000;

    w = Eigen::VectorXd::Zero(10);
    w << 0.0590618000000000, 0.0303744000000000, 0.2347740000000000,
      0.2466770000000000, 0.7662550000000000, 0.7523040000000000,
      0.0563272000000000, 0.4361250000000000, 0.8183180000000000,
      0.4684810000000000;

    cout << "mu = " << endl << model.mu << endl;
    std::cout << "Model pc = " << std::endl << model.pc << std::endl;

    cout << "xyn: " << endl << xyn << endl;

    cout << "w = " << endl << w << endl;

    TIME_TIC;
    res = optimizeStructureFromProjection(xyn, model, w);
    TIME_TOC("Full projection structure optimization with pc = []");

    std::cout << "Optimization result: R = " << std::endl << res.R << std::endl;
    cout << "t = " << res.t.transpose() << endl;
    cout << " C = " << res.C.transpose() << endl;

    cout << "Z = " << endl << res.Z.transpose() << endl << endl;

    TIME_TIC;
    res = optimizeStructureFromProjectionCoordinateDescent(xyn, model, w);
    TIME_TOC(
      "Full projection structure optimization COORDINATE DESCENT with pc = []");

    std::cout << "Optimization result: R = " << std::endl << res.R << std::endl;
    cout << "t = " << res.t.transpose() << endl;
    cout << " C = " << res.C.transpose() << endl;

    cout << "Z = " << endl << res.Z.transpose() << endl;

    // cout << "CERES!! SAME " << endl;

    // TIME_TIC;
    // resc = geometry_ceres::optimizeStructureFromProjection(xyn, model, w);
    // TIME_TOC("Full projection structure optimization CERES with pc = []");

    // std::cout << "Optimization result: R = " << std::endl << res.R <<
    // std::endl; cout << "t = " << res.t.transpose() << endl; cout << " C = "
    // << res.C.transpose() << endl;

    // cout << "Z = " << endl << res.Z.transpose() << endl << endl;
}