
#include "semantic_slam/CameraCalibration.h"

CameraCalibration::CameraCalibration(double fx, double fy, double s, double u0, double v0, double k1, 
            double k2, double p1, double p2)
    : fx_(fx), fy_(fy), s_(s), u0_(u0), v0_(v0), k1_(k1), k2_(k2), p1_(p1), p2_(p2)
{

}

Eigen::Vector2d CameraCalibration::uncalibrate(const Eigen::Vector2d& p) const
{
    // Code borrowed from opencv's projectpoints function
    double x = p(0), y = p(1);
    double r2 = x*x + y*y;
    double r4 = r2 * r2;

    double a1 = 2*x*y;
    double a2 = r2 + 2*x*x;
    double a3 = r2 + 2*y*y;

    double cdist = 1 + k1_*r2 + k2_*r4;

    double xd0 = x*cdist + p1_*a1 + p2_*a2;
    double yd0 = y*cdist + p1_*a3 + p2_*a1;

    return Eigen::Vector2d( fx_*xd0 + s_*yd0 + u0_, 
                            fy_*yd0 + v0_ );
}

Eigen::Vector2d CameraCalibration::calibrate(const Eigen::Vector2d& p) const
{
    // code borrowed from opencv's undistortpoints function

    int iters = 10; // opencv has this = 5?

    double u = p(0), v = p(1);

    // Initial guess without accounting for distortion
    double x = (u - u0_ - (s_/fy_)*(v - v0_)) / fx_;
    double y = (v - v0_) / fy_;

    double x0 = x;
    double y0 = y;

    for (int j = 0; j < iters; ++j) {
        double r2 = x*x + y*y;
        double icdist = 1.0 / (1 + (k2_*r2 + k1_)*r2);
        double deltaX = 2*p1_*x*y + p2_*(r2 + 2*x*x);
        double deltaY = p1_*(r2 + 2*y*y) + 2*p2_*x*y;
        x = (x0 - deltaX)*icdist;
        y = (y0 - deltaY)*icdist;
    }

    return Eigen::Vector2d(x,y);
}