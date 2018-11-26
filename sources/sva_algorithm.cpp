#include "sva_algorithm.h"

#include "containers.h"
#include "utils.h"

#include<iostream>
#include<vector>
#include <stdexcept>


// std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> sva::svaDL_cythonInterface(const Eigen::Map<Eigen::MatrixXd> &Y, int nbIt, double lbd_reg1, double lbd_reg2) {

// 	sva::DictAndCoef *dictchain = sva::svaDL(Y, nbIt, lbd_reg1, lbd_reg2);

// 	Eigen::MatrixXd dic;
// 	Eigen::MatrixXd coefs;

// 	dictchain->chainToMatrix(dic, coefs);

// 	delete dictchain;

// 	return std::make_tuple(dic, coefs);
// }


sva::DictAndCoef* sva::svaDL(const Eigen::MatrixXd &Y, int nbIt, double lbd_reg1, double lbd_reg2) {
	int D = Y.rows();
	int N = Y.cols();

	// Monitoring stuff
	Eigen::VectorXd monitor_K(nbIt);

	// Algorithm
	sva::DictAndCoef *dictCoef = new sva::DictAndCoef(D, N);

	for (int t = 0; t < nbIt; ++t) {
		
		// OMP step
		sva::_update_W_omp(Y, *dictCoef, lbd_reg1, lbd_reg2);

		// Update D
		double var_noise = 0; //np.var( mat_Y - np.dot(mat_D, mat_W) );
		sva::_update_D(Y, *dictCoef, var_noise);

		// Saving
		monitor_K[t] = dictCoef->get_K();
	}
	// We learn again the coefficients
	sva::_update_W_omp(Y, *dictCoef, lbd_reg1, lbd_reg2, false);

	return dictCoef;
}



void sva::_update_D(const Eigen::MatrixXd &cmatYobs, sva::DictAndCoef &dictAndCoef, double varnoise) {
	int iK = dictAndCoef.get_K();
	int iD = dictAndCoef.get_D();

	if (iK == 0)
		return;
	   
	for (int d = 0; d < iD; ++d) {
		
		Eigen::MatrixXd matBuf = Eigen::MatrixXd::Zero(iK, iK);
		Eigen::VectorXd rwDupd = Eigen::VectorXd::Zero(iK);


		// 1. buf : (WW^t + varnoise * K * I_K)^{-1}
		sva::Node* node1 = dictAndCoef.get_head();
		int irow = 0;
		while(node1) {
			int icol = 0;
			sva::Node* node2 = dictAndCoef.get_head();

			while(node2) {

				matBuf(irow, icol) = node1->vec_w.dot(node2->vec_w);
				matBuf(icol, irow) = matBuf(irow, icol); // by symetry

				node2 = node2->next;
				icol++;
			}

			if (varnoise > 0)
				matBuf(irow, irow) += varnoise * double(iK);

			node1 = node1->next;
			irow++;
		}

		// 2. Y[d, :] W^t stored in rwDupd	
		int icol = 0;
		node1 = dictAndCoef.get_head();
		while(node1) {
			rwDupd(icol) = cmatYobs.row(d) * node1->vec_w;

			node1 = node1->next;
			icol++;
		}

		// 3. Dnew[d, :] = Y[d, :] W^t (WW^t + varnoise * K * I_K)^{-1}
		rwDupd = rwDupd.transpose() * matBuf.inverse();

		// 4. Copy
		sva::Node* nodeD = dictAndCoef.get_head();
		icol = 0;
		while(nodeD) {
			nodeD->vec_d(d) = rwDupd(icol);
 			nodeD = nodeD->next;
 			icol++;
		}
	}
}


void sva::_update_W_omp(const Eigen::MatrixXd &Yobs, sva::DictAndCoef &dictAndCoef, double lbd_reg1, double lbd_reg2, bool addAtom) {
	int K = dictAndCoef.get_K();
	int D = dictAndCoef.get_D();
	int N = dictAndCoef.get_N();
	const double diffregu = lbd_reg1 - lbd_reg2;
	const sva::Node* node = nullptr;

	Eigen::VectorXd buf = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd Dw_n = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd residual = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd residual_n;

	// Compute average error
	node = dictAndCoef.get_head();
	residual = Yobs.rowwise().sum();
		while (node) {
			for (int n = 0; n < N; ++n)
				residual -= node->vec_w(n) * node->vec_d;

		node = node->next;
		}

	for (int n = 0; n < N; ++n) {

		/*****************************************
		 *
		 *			1. Performs OMP
		 *
		*****************************************/

		// update residual
		dictAndCoef.computeDWn(n, Dw_n);
		residual += Dw_n;

		dictAndCoef.setCoefToZero(n);

		double current_err = Yobs.col(n).norm();
		current_err *= current_err;
		double new_err = 0.;

		int sparsityLevel = 0;
		while (K > 0) { // if K == 0 this step is avoided

			// // 1.1 Add coefficient
			Eigen::VectorXd new_w;
			int newk = sva::utils::it_omp(Yobs.col(n), dictAndCoef, new_w, sparsityLevel, n);

				// compute new residu
			residual_n = Yobs.col(n);
			node = dictAndCoef.get_head();
			for (int k = 0; k < K; ++k) {
				residual_n -= new_w(k) * node->vec_d;
				node = node->next;		
			}

			new_err = residual_n.norm();
			new_err *= new_err;

			// 1.2 Test whether it is relevant to add it
			double a1 = new_err + lbd_reg2 * (double(sparsityLevel) + 1.);
			double a0 = current_err + lbd_reg2 * double(sparsityLevel);

			if ( (sparsityLevel == 0) && (dictAndCoef.get_mk(newk) == 0) )
				a1 += diffregu;

			// 1.3 Accepted or not?
			if (a1 < a0) {
				// 1.3.1 Accept, add and continue
				dictAndCoef.setCoef(new_w, n);
				current_err = new_err;
				sparsityLevel += 1;
			}
			else {
				// 1.3.2 Reject, stop
				break;
			}

			if (sparsityLevel == K)
				break;
		}

		// update residual
		dictAndCoef.computeDWn(n, Dw_n);
		residual -= Dw_n;

		/*****************************************
		 *
		 *		2. Delete unused feature
		 *
		*****************************************/

		dictAndCoef.clean_unuse_node();
		K = dictAndCoef.get_K();

		/*****************************************
		 *
		 *	3. Add a new feature using residual error
		 *
		*****************************************/
		if (addAtom) {

			double err_before = (Yobs.col(n) - Dw_n).norm();
			err_before *= err_before;

			// buf contain dnew
			if (residual.norm() > 0)
				buf = residual / residual.norm();
			else
				buf = Eigen::VectorXd::Zero(D);
			double coefnew = (Yobs.col(n) - Dw_n).dot(buf);
			double err_after = (Yobs.col(n) - Dw_n - coefnew * buf).norm();
			err_after *= err_after;


			//2.1 Accept or reject?
			double a1 = err_after + diffregu * (double(K) + 1.) + lbd_reg2 * (sparsityLevel + 1.);
			double a0 = err_before + diffregu * double(K) + lbd_reg2 * sparsityLevel;

			if (a1 < a0) {
				
				// Add feature
				dictAndCoef.add(buf, n, coefnew);
				K++;

				// Recompute coefficients with LS
				Eigen::VectorXi support_w(sparsityLevel + 1);
				dictAndCoef.get_support(n, support_w);
				sva::utils::solveLeastSquarePb(Yobs, dictAndCoef, support_w, n);

				// Update residu
				dictAndCoef.computeDWn(n, Dw_n);
				residual -= Dw_n;
			}
		}
	}
}
