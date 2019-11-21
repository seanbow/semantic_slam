#ifndef NISTER_COEFFS_H
#define NISTER_COEFFS_H
//#include "../../Eigen/SVD"

/*
enum
{
	coef_xxx,
	coef_xxy,
	coef_xyy,
	coef_yyy,
	coef_xxz,
	coef_xyz,
	coef_yyz,
	coef_xzz,
	coef_yzz,
	coef_zzz,
	coef_xx,
	coef_xy,
	coef_yy,
	coef_xz,
	coef_yz,
	coef_zz,
	coef_x,
	coef_y,
	coef_z,
	coef_1
};
*/

static constexpr int coef_xxx = 0;
static constexpr int coef_xxy = 1;
static constexpr int coef_xyy = 2;
static constexpr int coef_yyy = 3;
static constexpr int coef_xxz = 4;
static constexpr int coef_xyz = 5;
static constexpr int coef_yyz = 6;
static constexpr int coef_xzz = 7;
static constexpr int coef_yzz = 8;
static constexpr int coef_zzz = 9;
static constexpr int coef_xx = 10;
static constexpr int coef_xy = 11;
static constexpr int coef_yy = 12;
static constexpr int coef_xz = 13;
static constexpr int coef_yz = 14;
static constexpr int coef_zz = 15;
static constexpr int coef_x = 16;
static constexpr int coef_y = 17;
static constexpr int coef_z = 18;
static constexpr int coef_1 = 19;

inline Eigen::Matrix<double, 20, 1> o1(const Eigen::Matrix<double, 20, 1> &a, 
									   const Eigen::Matrix<double, 20, 1> &b)
{
	Eigen::Matrix<double, 20, 1> res = Eigen::Matrix<double, 20, 1>::Zero();

	res(coef_xx) = a(coef_x) * b(coef_x);
	res(coef_xy) = a(coef_x) * b(coef_y) + a(coef_y) * b(coef_x);
	res(coef_xz) = a(coef_x) * b(coef_z) + a(coef_z) * b(coef_x);
	res(coef_yy) = a(coef_y) * b(coef_y);
	res(coef_yz) = a(coef_y) * b(coef_z) + a(coef_z) * b(coef_y);
	res(coef_zz) = a(coef_z) * b(coef_z);
	res(coef_x) = a(coef_x) * b(coef_1) + a(coef_1) * b(coef_x);
	res(coef_y) = a(coef_y) * b(coef_1) + a(coef_1) * b(coef_y);
	res(coef_z) = a(coef_z) * b(coef_1) + a(coef_1) * b(coef_z);
	res(coef_1) = a(coef_1) * b(coef_1);

	return res;
}

inline Eigen::Matrix<double, 20, 1> o2(const Eigen::Matrix<double, 20, 1> &a,
									   const Eigen::Matrix<double, 20, 1> &b)
{
	Eigen::Matrix<double, 20, 1> res;

	res(coef_xxx) = a(coef_xx) * b(coef_x);
	res(coef_xxy) = a(coef_xx) * b(coef_y) + a(coef_xy) * b(coef_x);
	res(coef_xxz) = a(coef_xx) * b(coef_z) + a(coef_xz) * b(coef_x);
	res(coef_xyy) = a(coef_xy) * b(coef_y) + a(coef_yy) * b(coef_x);
	res(coef_xyz) = a(coef_xy) * b(coef_z) + a(coef_yz) * b(coef_x)
			+ a(coef_xz) * b(coef_y);
	res(coef_xzz) = a(coef_xz) * b(coef_z) + a(coef_zz) * b(coef_x);
	res(coef_yyy) = a(coef_yy) * b(coef_y);
	res(coef_yyz) = a(coef_yy) * b(coef_z) + a(coef_yz) * b(coef_y);
	res(coef_yzz) = a(coef_yz) * b(coef_z) + a(coef_zz) * b(coef_y);
	res(coef_zzz) = a(coef_zz) * b(coef_z);
	res(coef_xx) = a(coef_xx) * b(coef_1) + a(coef_x) * b(coef_x);
	res(coef_xy) = a(coef_xy) * b(coef_1) + a(coef_x) * b(coef_y) + a(coef_y)
			* b(coef_x);
	res(coef_xz) = a(coef_xz) * b(coef_1) + a(coef_x) * b(coef_z) + a(coef_z)
			* b(coef_x);
	res(coef_yy) = a(coef_yy) * b(coef_1) + a(coef_y) * b(coef_y);
	res(coef_yz) = a(coef_yz) * b(coef_1) + a(coef_y) * b(coef_z) + a(coef_z)
			* b(coef_y);
	res(coef_zz) = a(coef_zz) * b(coef_1) + a(coef_z) * b(coef_z);
	res(coef_x) = a(coef_x) * b(coef_1) + a(coef_1) * b(coef_x);
	res(coef_y) = a(coef_y) * b(coef_1) + a(coef_1) * b(coef_y);
	res(coef_z) = a(coef_z) * b(coef_1) + a(coef_1) * b(coef_z);
	res(coef_1) = a(coef_1) * b(coef_1);

	return res;
}
#endif // NISTER_COEFFS_H
