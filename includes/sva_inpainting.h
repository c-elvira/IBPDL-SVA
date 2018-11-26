#pragma once

//#include <tuple>
#include "containers.h"
#include "typedef.h"

#include "Eigen/Dense"


namespace sva {

	/**
	 * @brief [brief description]
	 * @details         
	 * Solve ...

	        Input:
	            - mat_Y, the PxN observation matrix
	            - nbIt, an integer containing the number of observation
	            - lbd_reg1, a positive real number
	            - lbd_reg2, a positive real number, optional (default is 0)
	            - updt_W, default is 'gibbs', can be 'OMP'
	            - doptions, a dictionnary containing options (see below)


	        Output:
	            - mat_D, the PxK inferred dictionary
	            - mat_W, the KxN (sparse) matrix of (real) coefficient
	            - mon_K, a nbIt array containing the value of K at each iteration
	            - mon_error, a nbIt array containing the value of K at each iteration


	        Keys for doptions are:
	            * verbose, True or False
	            * saveIt, True or False
	            * saveItFolder, a string
	            * eraseIt, True or False
	            * newAtome, averageErr or residue
	            * updt_W, Gibbs, OMP or MP

	        Copy paste this example for a default set-up
	        >>> doptions = {'verbose':True, 'newAtome':'averageErr', 'updt_W':OMP, 'updt_D'= 'withoutNoise'}
	 * 
	 * @param Y [description]
	 * @param nbIt [description]
	 * @param lbd_reg1 [description]
	 * @param lbd_reg2 [description]
	 */ 
	sva::DictAndCoef* svaDLinpainting(const Eigen::MatrixXd &cmatYobs, const Eigen::custom::MatrixXb &cmatMask,  int nbIt, double lbd_reg1, double lbd_reg2);


	namespace inpainting {

		/**
		 * @brief [brief description]
		 * @details 
		 *  Inputs:
	            - mat_Y, the PxN observation array
	            - mat_W, the KxN current coefficient array

	        Outputs:
	            - mat_D, the new dictionary
		 * 
		 * @param Y [description]
		 * @param dictAndCoef [description]
		 * @param e [description]
		 */
		void _update_D(const Eigen::MatrixXd &cmatYobs, const Eigen::custom::MatrixXb &cmatMask, sva::DictAndCoef &dictAndCoef, double varnoise);

		void _update_W_omp(const Eigen::MatrixXd &cmatYobs, const Eigen::custom::MatrixXb &matMask, sva::DictAndCoef &dictAndCoef, double lbd_reg1, double lbd_reg2, bool addAtom=true);
	}
}