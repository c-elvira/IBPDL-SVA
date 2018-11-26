#include "sva_inpainting.h"

#include "containers.h"
#include "utils.h"

#include<iostream>
#include<vector>
#include <stdexcept>


sva::DictAndCoef* sva::svaDLinpainting(const Eigen::MatrixXd &cmatYobs, const Eigen::custom::MatrixXb &cmatMask, int nbIt, double lbd_reg1, double lbd_reg2) {

	int iD = cmatYobs.rows();
	int iN = cmatYobs.cols();

	// Monitoring stuff
	Eigen::VectorXd monitor_K(nbIt);

	// Algorithm
	sva::DictAndCoef *dictCoef = new sva::DictAndCoef(iD, iN);

	for (int t = 0; t < nbIt; ++t) {
		
		// OMP step
		sva::inpainting::_update_W_omp(cmatYobs, cmatMask, *dictCoef, lbd_reg1, lbd_reg2);

		// Update D
		double var_noise = 0; //np.var( mat_Y - np.dot(mat_D, mat_W) );
		sva::inpainting::_update_D(cmatYobs, cmatMask, *dictCoef, var_noise);

		// Saving
		monitor_K[t] = dictCoef->get_K();
	}
	// We learn again the coefficients
	sva::inpainting::_update_W_omp(cmatYobs, cmatMask, *dictCoef, lbd_reg1, lbd_reg2, false);

	return dictCoef;
}


void sva::inpainting::_update_D(const Eigen::MatrixXd &cmatYobs, const Eigen::custom::MatrixXb &cmatMask, sva::DictAndCoef &dictAndCoef, double varnoise) {

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

				matBuf(irow, icol) = node1->vec_w.dot( node2->vec_w.cwiseProduct(cmatMask.row(d).transpose().cast<double>()) );
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


void sva::inpainting::_update_W_omp(const Eigen::MatrixXd &cmatYobs, const Eigen::custom::MatrixXb &cmatMask, sva::DictAndCoef &dictAndCoef, double lbd_reg1, double lbd_reg2, bool addAtom) {

	int iK = dictAndCoef.get_K();
	int iD = dictAndCoef.get_D();
	int iN = dictAndCoef.get_N();
	const double diffregu = lbd_reg1 - lbd_reg2;
	const sva::Node* node = nullptr;

	Eigen::VectorXd buf = Eigen::VectorXd::Zero(iD);
	Eigen::VectorXd HDw_n = Eigen::VectorXd::Zero(iD);
	Eigen::VectorXd residual = Eigen::VectorXd::Zero(iD);
	Eigen::VectorXd residual_n;

	// Compute average error
	node = dictAndCoef.get_head();
	
	// residual = sum_n Mask[:, n] .* Y[:, n]
	residual = cmatYobs.rowwise().sum();

	while (node) {
		for (int n = 0; n < iN; ++n)
			residual -= node->vec_w(n) * node->vec_d.cwiseProduct(cmatMask.col(n).cast<double>());

		node = node->next;
	}

	for (int n = 0; n < iN; ++n) {

		/*****************************************
		 *
		 *			1. Performs OMP
		 *
		*****************************************/

		// update residual
		dictAndCoef.computeDHWn(n, cmatMask.col(n), HDw_n);
		residual += HDw_n;

		dictAndCoef.setCoefToZero(n);

		double current_err = cmatYobs.col(n).norm();
		current_err *= current_err;
		double new_err = 0.;

		int sparsityLevel = 0;
		while (iK > 0) { // if K == 0 this step is avoided

			// // 1.1 Add coefficient
			Eigen::VectorXd new_w;
			int newk = sva::utils::it_ompMasked(cmatYobs.col(n), cmatMask.col(n), dictAndCoef, new_w, sparsityLevel, n);

				// compute new residu
			residual_n = cmatYobs.col(n);
			node = dictAndCoef.get_head();
			for (int k = 0; k < iK; ++k) {
				residual_n -= new_w(k) * node->vec_d.cwiseProduct(cmatMask.col(n).cast<double>());
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

			if (sparsityLevel == iK)
				break;
		}

		// update residual
		dictAndCoef.computeDHWn(n, cmatMask.col(n), HDw_n);
		residual -= HDw_n;


		/*****************************************
		 *
		 *		2. Delete unused feature
		 *
		*****************************************/

		dictAndCoef.clean_unuse_node();
		iK = dictAndCoef.get_K();



		/*****************************************
		 *
		 *	3. Add a new feature using residual error
		 *
		*****************************************/
		if (addAtom) {

			double err_before = (cmatYobs.col(n) - HDw_n).norm();
			err_before *= err_before;

			// buf contain dnew
			if (residual.norm() > 0)
				buf = residual / residual.norm();
			else
				buf = Eigen::VectorXd::Zero(iD);
			double coefnew = (cmatYobs.col(n) - HDw_n).dot(buf);
			double err_after = (cmatYobs.col(n) - HDw_n - coefnew * buf.cwiseProduct(cmatMask.col(n).cast<double>())).norm();
			err_after *= err_after;


			//2.1 Accept or reject?
			double a1 = err_after + diffregu * (double(iK) + 1.) + lbd_reg2 * (sparsityLevel + 1.);
			double a0 = err_before + diffregu * double(iK) + lbd_reg2 * sparsityLevel;

			if (a1 < a0) {
				
				// Add feature
				dictAndCoef.add(buf, n, coefnew);
				iK++;

				// Recompute coefficients with LS
				Eigen::VectorXi support_w(sparsityLevel + 1);
				dictAndCoef.get_support(n, support_w);
				sva::utils::solveMaskedLeastSquarePb(cmatYobs.col(n), cmatMask.col(n), dictAndCoef, support_w, n);

				// Update residu
				dictAndCoef.computeDHWn(n, cmatMask.col(n), HDw_n);
				residual -= HDw_n;
			}
		}
	}
}

