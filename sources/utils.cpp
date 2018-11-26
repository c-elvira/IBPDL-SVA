#include "utils.h"

#include <cmath>
#include <iostream>

/**
 * @brief return the l2 norm
 * @details return the l2 norm given by
 * 	|| y - Dw ||_2
 * 
 * @param y [description]
 * @param dictAndCoef [description]
 * @param n [description]
 * @return [description]
 */
double sva::utils::compute_l2_error(const Eigen::VectorXd& y, const sva::DictAndCoef &dictAndCoef, int n) {
	// Buffer
	Eigen::VectorXd buf = y;

	// Loop
	const sva::Node* node = dictAndCoef.get_head();
	while(node) {
		buf -= node->vec_w(n) * node->vec_d;
		node = node->next;
	}

	return buf.norm();
}


/**
 * @brief Perform one additional iteration of OMP from vec_s
 * @details 
 * (Try to) solve
 * $$ \arg\min_{s} ||y - mat_D s|| s.t. supp(s) \in supp(vec_s),  || s ||_0 = |supp(vec_s)|+1 $$
 * 
 * @param y the Px1 observation array
 * @param D the PxK dictionary array
 * @param s the output of previous OMP iteration
 * @param t is the current iteration number
 * @return the Kx1 coefficient array
 */
int sva::utils::it_omp(const Eigen::VectorXd &y, const sva::DictAndCoef &dictAndCoef, Eigen::VectorXd &new_w, int t, int n) {
	int D = dictAndCoef.get_D();
	int K = dictAndCoef.get_K();
	int k = 0;
	int i = 0;

	if (D != y.size())
		throw std::invalid_argument( "it_omp: size does not match" );

	// To do
	Eigen::MatrixXd	feats(D, t+1);
	Eigen::VectorXi support_w(t+1);
	Eigen::VectorXd buf_coef = Eigen::VectorXd::Zero(t+1);
	Eigen::VectorXd resisdual;
	new_w = Eigen::VectorXd::Zero(K);

	k = 0;
	i = 0;
	resisdual = y;
	const sva::Node* node = dictAndCoef.get_head();
	while(node) {

		if (node->vec_w(n) != 0){
			feats.col(k) = node->vec_d;
			new_w(i) = node->vec_w(n);
			
			// residual vector
			resisdual -= new_w(i) * node->vec_d;

			support_w(k) = i;
			k++;
		}

		node = node->next;
		i++;
	}

	if (k != t)
		throw std::invalid_argument( "it_omp: support and sparsity do not match" );

	// 1. select the more correlated atom
	node = dictAndCoef.get_head();
	i = 0;
	k = -1;
	double max_sp = -1.;
	node = dictAndCoef.get_head();
	while(node) {
		double sp = std::abs( resisdual.dot(node->vec_d) );

		if (sp > max_sp) {
			max_sp = sp;
			k = i;
		}

		node = node->next;
		i++;
	}

	// 2. Allocation
	support_w(t) = k;
	dictAndCoef.get_featk(k, feats, t);

	// 3. Perform Least Square
	buf_coef = feats.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

	// 4. Allocate solution
	for (int k = 0; k < t+1; ++k)
		new_w( support_w(k) ) = buf_coef(k);

	return k;
}


double sva::utils::computeCoherence(const Eigen::MatrixXd& dic) {
	//int D = dic.rows();
	int K = dic.cols();

	double mu = 0;
	for (int k1 = 0; k1 < K; ++k1) {
		for (int k2 = k1+1; k2 < K; ++k2) {

			double buf = std::abs( dic.col(k1).dot(dic.col(k2)) );
			mu = buf > mu ? buf : mu;
		}
	}

	return mu;
}


void sva::utils::solveLeastSquarePb(const Eigen::MatrixXd& yobs, sva::DictAndCoef &dictAndCoef, const Eigen::VectorXi &support, int n) {
	int sparsityLevel = support.size();
	int D = yobs.rows();

	Eigen::MatrixXd matD(D, sparsityLevel);
	Eigen::VectorXd vecw = Eigen::VectorXd::Zero(sparsityLevel);

	sva::Node* node = nullptr;

	// Create dictionnary
	int i = 0;
	int k = 0;
	node = dictAndCoef.get_head();
	while (node) {
		
		if ( support(k) == i ) {
			matD.col(k) = node->vec_d;
			k++;

			if (k == sparsityLevel)
				break;
		}

		node = node->next;
		i++;
	}

	// Solve least Square Pb
	vecw = matD.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(yobs.col(n));

	// Copy
	dictAndCoef.setCoefToZero(n);

	node = dictAndCoef.get_head();
	k = 0;
	i = 0;
	while (node) {
		
		if ( support(k) == i ) {
			node->vec_w(n) = vecw(k);
			node->mk++;
			k++;

			if (k == sparsityLevel)
				break;
		}

		node = node->next;
		i++;
	}
}

/**
 * 
 * 	Inpainting tools
 * 
 */

int sva::utils::it_ompMasked(const Eigen::VectorXd &cvecYobs,  const Eigen::custom::VectorXb& cvecMask, const sva::DictAndCoef &dictAndCoef, Eigen::VectorXd &new_w, int t, int n) {
	int iD = dictAndCoef.get_D();
	int iK = dictAndCoef.get_K();
	int k = 0;
	int i = 0;

	if (iD != cvecYobs.size())
		throw std::invalid_argument( "it_omp: size does not match" );

	// To do
	Eigen::MatrixXd	feats(iD, t+1);
	Eigen::VectorXi support_w(t+1);
	Eigen::VectorXd buf_coef = Eigen::VectorXd::Zero(t+1);
	Eigen::VectorXd resisdual;
	new_w = Eigen::VectorXd::Zero(iK);

	k = 0;
	i = 0;
	resisdual = cvecYobs;
	const sva::Node* node = dictAndCoef.get_head();
	while(node) {

		if (node->vec_w(n) != 0){
			feats.col(k) = cvecMask.cast<double>().cwiseProduct(node->vec_d);
			new_w(i) = node->vec_w(n);
			
			// residual vector
			resisdual -= new_w(i) * cvecMask.cast<double>().cwiseProduct(node->vec_d);

			support_w(k) = i;
			k++;
		}

		node = node->next;
		i++;
	}

	if (k != t)
		throw std::invalid_argument( "it_omp: support and sparsity do not match" );

	// 1. select the more correlated atom
	node = dictAndCoef.get_head();
	i = 0;
	k = -1;
	double max_sp = -1.;
	node = dictAndCoef.get_head();
	while(node) {
		double sp = std::abs( resisdual.dot(cvecMask.cast<double>().cwiseProduct(node->vec_d)) );

		if (sp > max_sp) {
			max_sp = sp;
			k = i;
		}

		node = node->next;
		i++;
	}

	// 2. Allocation
	support_w(t) = k;
	dictAndCoef.get_featk(k, feats, t);

	// 3. Perform Least Square
	buf_coef = feats.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cvecYobs);

	// 4. Allocate solution
	for (int k = 0; k < t+1; ++k)
		new_w( support_w(k) ) = buf_coef(k);

	return k;
}


void sva::utils::solveMaskedLeastSquarePb(const Eigen::VectorXd& cvecYobs, const Eigen::custom::VectorXb& cvecMask, sva::DictAndCoef &dictAndCoef, const Eigen::VectorXi &support, int n) {
	int sparsityLevel = support.size();
	int D = cvecYobs.size();

	Eigen::MatrixXd matD(D, sparsityLevel);
	Eigen::VectorXd vecw = Eigen::VectorXd::Zero(sparsityLevel);

	sva::Node* node = nullptr;

	// Create dictionnary
	int i = 0;
	int k = 0;
	node = dictAndCoef.get_head();
	while (node) {
		
		if ( support(k) == i ) {
			matD.col(k) = cvecMask.cast<double>().cwiseProduct(node->vec_d);
			k++;

			if (k == sparsityLevel)
				break;
		}

		node = node->next;
		i++;
	}

	// Solve least Square Pb
	vecw = matD.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cvecYobs);

	// Copy
	dictAndCoef.setCoefToZero(n);

	node = dictAndCoef.get_head();
	k = 0;
	i = 0;
	while (node) {
		
		if ( support(k) == i ) {
			node->vec_w(n) = vecw(k);
			node->mk++;
			k++;

			if (k == sparsityLevel)
				break;
		}

		node = node->next;
		i++;
	}
}
