#include "cythonInterface.h"

#include <iostream>

#include "sva_algorithm.h"
#include "sva_inpainting.h"


void SvaDl_cythonInterface::run(const RowMajorArrayMap &Y, int nbIt, double lbd_reg1, double lbd_reg2) {

	sva::DictAndCoef *dictchain = sva::svaDL(Y, nbIt, lbd_reg1, lbd_reg2);

	dictchain->chainToMatrix(this->dic, this->coefs);
	delete dictchain;
}

void SvaDlInpainting_cythonInterface::run(const RowMajorArrayMap &Y, const RowMajorBoolArrayMap &cmatMask, int nbIt, double lbd_reg1, double lbd_reg2) {

	sva::DictAndCoef *dictchain = sva::svaDLinpainting(Y, cmatMask.cast<bool>(), nbIt, lbd_reg1, lbd_reg2);

	dictchain->chainToMatrix(this->dic, this->coefs);
	delete dictchain;
}