#pragma once

#include <exception>

#include "Eigen/Dense"

#include "containers.h"
#include "typedef.h"

// See below for the implementation
namespace sva {

	namespace utils {

		double compute_l2_error(const Eigen::VectorXd& y, const sva::DictAndCoef &dictAndCoef, int n);

		int it_omp(const Eigen::VectorXd &vec_y, const sva::DictAndCoef &dictAndCoef, Eigen::VectorXd &new_w, int t, int n);

		double computeCoherence(const Eigen::MatrixXd& dic);

		void solveLeastSquarePb(const Eigen::MatrixXd& yobs, sva::DictAndCoef &dictAndCoef, const Eigen::VectorXi &support, int n);


		int it_ompMasked(const Eigen::VectorXd &cvecYobs, const Eigen::custom::VectorXb& cvecMask, const sva::DictAndCoef &dictAndCoef, Eigen::VectorXd &new_w, int t, int n);

		void solveMaskedLeastSquarePb(const Eigen::VectorXd& cvecYobs, const Eigen::custom::VectorXb& cvecMask, sva::DictAndCoef &dictAndCoef, const Eigen::VectorXi &support, int n);
	}
}
