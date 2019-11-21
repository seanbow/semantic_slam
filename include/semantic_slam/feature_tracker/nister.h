#ifndef NISTER_H
#define NISTER_H
//#include "../../Eigen/Core"
//#include "../numerics.h"
#include <eigen3/Eigen/Eigenvalues>
//#include "Eigen/Core"
#include <eigen3/Eigen/SVD>
#include "nister_coeffs.h"
/* Nister's Five point algorithm simple implementation.
 As implemented in the Matlab code at his website and
 at the implementation in the src code of the libmv project.*/

inline void fivePoint(const Eigen::Matrix<double, 3, Eigen::Dynamic> &p1,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic> &p2,
		      const Eigen::Matrix<int,5,1> &set_,
		      Eigen::Matrix<double, 3, 30>* essential,
		      int* numOfSols)
{
	*numOfSols = 0;

	Eigen::Matrix<double, 9, 9> constraint;
	constraint.setZero();
	for (int i = 0; i < 5; i++)
	{
		constraint(i, 0) = (p2)(0, set_(i,0)) * (p1)(0, set_(i,0));
		constraint(i, 1) = (p2)(0, set_(i,0)) * (p1)(1, set_(i,0));
		constraint(i, 2) = (p2)(0, set_(i,0));
		constraint(i, 3) = (p2)(1, set_(i,0)) * (p1)(0, set_(i,0));
		constraint(i, 4) = (p2)(1, set_(i,0)) * (p1)(1, set_(i,0));
		constraint(i, 5) = (p2)(1, set_(i,0));
		constraint(i, 6) = (p1)(0, set_(i,0));
		constraint(i, 7) = (p1)(1, set_(i,0));
		constraint(i, 8) = 1.0;
	}
	Eigen::Matrix<double, 9, 4> nullspaceOfConstraint;

	//nullspaceOfConstraint = constraint.svd().matrixV().block(0,5,9,6);
	Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9> > svd(constraint, Eigen::ComputeFullV);
	nullspaceOfConstraint = svd.matrixV().topRightCorner<9, 4> ();
	//    Eigen::Vector4d test;
	//    test << rand()%100,rand()%100,rand()%100,rand()%100;
	//    assert(norm((constraint*nullspaceOfConstraint*test))<1e-8);


	Eigen::Matrix<double, 20, 1> E[3][3];
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			// std::cout << "i = " << i << ", j = " << j << std::endl;
			// std::cout << "size(E[i][j]) = " << E[i][j].rows() << " x " << E[i][j].cols() << std::endl;
			E[i][j] = Eigen::Matrix<double, 20, 1>::Zero();
			(E[i][j])(coef_x, 0) = nullspaceOfConstraint(3 * i + j, 0);
			(E[i][j])(coef_y, 0) = nullspaceOfConstraint(3 * i + j, 1);
			(E[i][j])(coef_z, 0) = nullspaceOfConstraint(3 * i + j, 2);
			(E[i][j])(coef_1, 0) = nullspaceOfConstraint(3 * i + j, 3);
		}
	}

	// The constraint matrix.
	Eigen::Matrix<double, 10, 20> M;
	int mrow = 0;

	// Determinant constraint det(E) = 0; equation (19) of Nister [2].
	M.row(mrow++) = o2(o1(E[0][1], E[1][2]) - o1(E[0][2], E[1][1]), E[2][0])
			+ o2(o1(E[0][2], E[1][0]) - o1(E[0][0], E[1][2]), E[2][1]) + o2(o1(
			E[0][0], E[1][1]) - o1(E[0][1], E[1][0]), E[2][2]);

	// Cubic singular values constraint.
	// Equation (20).
	Eigen::Matrix<double, 20, 1> EET[3][3];
	for (int i = 0; i < 3; ++i)
	{ // Since EET is symmetric, we only compute
		for (int j = 0; j < 3; ++j)
		{ // its upper triangular part.
			if (i <= j)
			{
				EET[i][j] = o1(E[i][0], E[j][0]) + o1(E[i][1], E[j][1]) + o1(
						E[i][2], E[j][2]);
			}
			else
			{
				EET[i][j] = EET[j][i];
			}
		}
	}

	// Equation (21).
	Eigen::Matrix<double, 20, 1> (&L)[3][3] = EET;
	Eigen::Matrix<double, 20, 1> trace = 0.5 * (EET[0][0] + EET[1][1] + EET[2][2]);
	for (int i = 0; i < 3; ++i)
	{
		L[i][i] -= trace;
	}

	// Equation (23).
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			Eigen::Matrix<double, 20, 1> LEij = o2(L[i][0], E[0][j]) + o2(L[i][1], E[1][j])
					+ o2(L[i][2], E[2][j]);
			M.row(mrow++) = LEij;
		}
	}

	/*Gauss Jordan*/

	for (int i = 0; i < 10; ++i)
	{
		M.row(i) /= M(i, i);
		for (int j = i + 1; j < 10; ++j)
		{
			M.row(j) = M.row(j) / M(j, i) - M.row(i);
		}
	}

	// Backsubstitution.
	for (int i = 9; i >= 0; --i)
	{
		for (int j = 0; j < i; ++j)
		{
			M.row(j) = M.row(j) - M(j, i) * M.row(i);
		}
	}

	Eigen::Matrix<double, 10, 10> B = M.topRightCorner<10, 10> ();
	Eigen::Matrix<double, 10, 10> At = Eigen::Matrix<double, 10, 10>::Zero();
	At.row(0) = -B.row(0);
	At.row(1) = -B.row(1);
	At.row(2) = -B.row(2);
	At.row(3) = -B.row(4);
	At.row(4) = -B.row(5);
	At.row(5) = -B.row(7);
	At(6, 0) = 1;
	At(7, 1) = 1;
	At(8, 3) = 1;
	At(9, 6) = 1;

	// Compute solutions from action matrix's eigenvectors.
	Eigen::EigenSolver<Eigen::Matrix<double, 10, 10> > es(At);
	Eigen::Matrix<std::complex<double>, 10, 10> V = es.eigenvectors();
	Eigen::Matrix<std::complex<double>, 4, 10> SOLS;
	SOLS.row(0) = V.row(6).array() / V.row(9).array();
	SOLS.row(1) = V.row(7).array() / V.row(9).array();
	SOLS.row(2) = V.row(8).array() / V.row(9).array();
	SOLS.row(3).setOnes();

	// Get the ten candidate E matrices in vector form.
	Eigen::Matrix<std::complex<double>, 9, 4> nullspaceOfConstraint2;
	nullspaceOfConstraint2.imag().setZero();
	nullspaceOfConstraint2.real() = nullspaceOfConstraint;
	Eigen::Matrix<std::complex<double>, 9, 10> Evec = nullspaceOfConstraint2 * SOLS;

	// Build essential matrices for the real solutions.
	Eigen::Matrix<double, 3, 30> Es;
	for (int s = 0; s < 10; ++s)
	{
		Evec.col(s) /= Evec.col(s).norm();
		bool is_real = true;
		for (int i = 0; i < 9; ++i)
		{
			if (fabs(Evec(i, s).imag()) >= 1e-1)
			{
				is_real = false;
				break;
			}
		}
		if (is_real)
		{
			Eigen::Matrix3d E;
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					E(i, j) = Evec(3 * i + j, s).real();
				}
			}
			(*essential).block(0, (*numOfSols) * 3, 3, 3) = E;
			(*numOfSols)++;
			//         for (int i=0;i<5;i++)   {
			//             std::cout << s << " " << p2.col(i).transpose()*E*p1.col(i) <<  "cost\n";
			//         }
		}

	}
}
//Contraint coefficients from libmv source code and nister's matlab code:

#endif // NISTER_H
