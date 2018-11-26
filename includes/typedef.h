#pragma once

namespace Eigen {

	namespace custom {

		typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
		typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;
	}
}

/**
 * Additional stuff : Hungarian notation
 * 
 * Usual prefix
 * 	- i: integer. Ex: iD, iN, iK
 * 
 * Specific prefix
 * 	- mat: matrix. Ex: matY, matMask
 * 	- cmat: const matrix 
 */