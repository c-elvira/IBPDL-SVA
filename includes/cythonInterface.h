#pragma once

#include "containers.h"

#include "Eigen/Dense"

typedef Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > RowMajorArrayMap;
typedef Eigen::Map<Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > RowMajorBoolArrayMap;

class SvaDl_cythonInterface {
public:
    SvaDl_cythonInterface() {};
    ~SvaDl_cythonInterface() {};

	void run(const RowMajorArrayMap &Y, int nbIt, double lbd_reg1, double lbd_reg2);


    Eigen::MatrixXd &get_dic() { return this->dic; }
    Eigen::MatrixXd &get_coefs() {return this->coefs;}

private:
	Eigen::MatrixXd dic;
	Eigen::MatrixXd coefs;
};

class SvaDlInpainting_cythonInterface {
public:
    SvaDlInpainting_cythonInterface() {};
    ~SvaDlInpainting_cythonInterface() {};

	void run(const RowMajorArrayMap &Y, const RowMajorBoolArrayMap &cmatMask, int nbIt, double lbd_reg1, double lbd_reg2);


    Eigen::MatrixXd &get_dic() { return this->dic; }
    Eigen::MatrixXd &get_coefs() {return this->coefs;}

private:
	Eigen::MatrixXd dic;
	Eigen::MatrixXd coefs;
};