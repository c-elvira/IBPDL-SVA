#include "containers.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>

#include "Eigen/Dense"


sva::Node::Node(int D, int N) {

	this->vec_d.resize(D);
	this->vec_w.resize(N);

	this->mk = 0;
	this->next = nullptr;
}


sva::Node::Node(int D, int N, const Eigen::VectorXd &dic) : sva::Node::Node(D, N) {

	if (dic.size() != D )
		throw std::invalid_argument( "dic has size != D" );

	this->vec_d = Eigen::VectorXd(dic);
} 

sva::Node::Node(int D, int N, const Eigen::VectorXd &dic, const Eigen::VectorXd &coefs) : sva::Node::Node(D, N) {

	if (dic.size() != D )
		throw std::invalid_argument( "dic has size != D" );

	if (coefs.size() != N )
		throw std::invalid_argument( "coefs has size != N" );

	this->vec_d = Eigen::VectorXd(dic);
	this->vec_w = Eigen::VectorXd(coefs);

	for (int n = 0; n < N; ++n) {
		if (coefs(n) != 0) {
			this->mk++;
		}
	}
} 




sva::DictAndCoef::DictAndCoef(int D, int N) {

	this->D = D;
	this->N = N;

    this->length = 0;
    this->head = nullptr;
    this->last = nullptr;
}


sva::DictAndCoef::~DictAndCoef() {

	if (this->length > 0) {

		sva::Node* node = this->head;
		sva::Node* copy = nullptr;
    	while(node){
    		copy = node;
    		node = node->next;
    		
    		if (copy != nullptr)
	    		delete copy;
    	}
	}
}


void sva::DictAndCoef::add(const Eigen::VectorXd& new_d) {

	this->add(new_d, Eigen::VectorXd::Zero(this->N));
}


void sva::DictAndCoef::add(const Eigen::VectorXd& new_d, int n, double w) {

	Eigen::VectorXd vec_w = Eigen::VectorXd::Zero(this->N);
	vec_w(n) = w;

	this->add(new_d, vec_w);
}


void sva::DictAndCoef::add(const Eigen::VectorXd& new_d, const Eigen::VectorXd& new_coefs) {

	sva::Node *newNode = new sva::Node(this->D, this->N, new_d, new_coefs);

	if (this->length == 0) {

		this->head = newNode;
		this->last = newNode;
	}

	else {

		this->last->next = newNode;
		this->last = newNode;
	}

	this->length++;
}


void sva::DictAndCoef::save(std::string filename) {

	// Create format
	Eigen::IOFormat jsonFormat(Eigen::FullPrecision, 0, 
		", ", ",\n", "\t\t[", "]", "[\n", "\n\t]");

	std::ofstream file(filename);
	if (!file.is_open())
		throw std::invalid_argument("Problem with opening file while saving matrix");

	// 1. Create matrix
	Eigen::MatrixXd dict(this->D, this->length);
	Eigen::MatrixXd wcoef(this->length, this->N);

	unsigned int k = 0;
	sva::Node* node = this->head;
	while (node) {

		for (unsigned int d = 0; d < this->D; ++d)
			dict(d, k) = node->vec_d[d];

		for (unsigned int n = 0; n < this->N; ++n)
			wcoef(k, n) = node->vec_w[n];

		node = node->next;
		k++;
	}

	// 2. Save
	  if (file.is_open()) {
	
    	file << "{" << std::endl;
    	
    	file << "\t\"dim_D\": " << this->D << "," << std::endl;
    	file << "\t\"dim_N\": " << this->N << "," << std::endl;
    	file << "\t\"dim_K\": " << this->length << "," << std::endl;

    	file << "\t\"mat_D\": " << dict.format(jsonFormat) << "," << std::endl;
    	file << "\t\"mat_W\": " << wcoef.format(jsonFormat) << std::endl;

    	file << "}" << std::endl;
  	}
}


void sva::DictAndCoef::setCoefToZero(int n) {

	sva::Node* node = this->head;
	while (node) {
		
		if ( std::abs(node->vec_w[n]) > 0) {
			node->vec_w[n] = 0;
			node->mk--;
		}

		node = node->next;
	}	
}


void sva::DictAndCoef::setCoef(const Eigen::VectorXd &coefs, int n) {

	if (coefs.size() != this->length)
		throw std::invalid_argument( "setCoef: size does not match" );

	sva::Node* node = this->head;

	for (int k = 0; k < coefs.size(); ++k) {
		
		if (node->vec_w(n) != 0 ){
			if (coefs(k) == 0)
				node->mk--;
		}
		else {
			if (coefs(k) != 0)
				node->mk++;
		}

		node->vec_w(n) = coefs(k);
		node = node->next;
	}	
}


void sva::DictAndCoef::get_featk(int k, Eigen::MatrixXd &dic, int t) const {

	int i = 0;
	sva::Node* node = this->head;
	while (node) {
		
		if (i == k) {
			dic.col(t) = node->vec_d;
			return;
		}

		node = node->next;
		i++;
	}	
}


void sva::DictAndCoef::get_support(int n, Eigen::VectorXi &output) const{

	//int sparsityLevel = output.size();
	int i = 0;
	int k = 0;
	sva::Node* node = this->head;
	while (node) {
		
		if ( node->vec_w(n) != 0 ) {
			output(k) = i;
			k++;
		}

		node = node->next;
		i++;
	}		
}


void sva::DictAndCoef::computeDWn(int n, Eigen::VectorXd &output) const {

	output = Eigen::VectorXd::Zero(this->D);

	sva::Node* node = this->head;
	while (node) {
		output += node->vec_w(n) * node->vec_d;
		node = node->next;
	}	
}


void sva::DictAndCoef::computeDHWn(int n, const Eigen::custom::VectorXb& cvecMask, Eigen::VectorXd &vecOutput) const {

	vecOutput = Eigen::VectorXd::Zero(this->D);

	sva::Node* node = this->head;
	while (node) {
		vecOutput += node->vec_w(n) * cvecMask.cast<double>().cwiseProduct(node->vec_d);
		node = node->next;
	}	
}


void sva::DictAndCoef::print() {

    std::cout << "This dictionary contains " << this->length << " elements" << std::endl;
    
}


void sva::DictAndCoef::print_KcolOfD(int k) {

	if (k >= int(this->length))
		k = this->length;
	else if (k < 0)
		throw std::invalid_argument( "k should be = 0" );

	std::cout << "print the first " << k << " column of D" << std::endl;
	for (unsigned int d = 0; d < this->D; ++d) {
		
		sva::Node* node = this->head;
		int nb = 0;
    	while(nb < k){
    		std::cout << node->vec_d[d] << "\t";
    		node = node->next;
    		nb++;
    	}
	}
}


void sva::DictAndCoef::clean_unuse_node() {

	sva::Node* node = this->head;
	while (node) {

		if (node->mk == 0) {
			sva::Node* copy = node;
			node = node->next;
			delete copy;

			this->head = node;
			this->length--;
		}
		else
			node = node->next;
	}
}


unsigned int sva::DictAndCoef::get_mk(unsigned int k) const {

	sva::Node* node = this->head;
	for (unsigned int t = 0; t < this->length; ++t) {
		
		if (t == k)
			return node->mk;

		node = node->next;
	}

	return 0;
}


void sva::DictAndCoef::chainToMatrix(Eigen::MatrixXd &dic, Eigen::MatrixXd &coefs) const {

	dic.resize(this->D, this->length);
	coefs.resize(this->length, this->N);

	sva::Node* node = this->head;
	for (unsigned int k = 0; k < this->length; ++k) {
		
		for (unsigned int d = 0; d < this->D; ++d)
			dic(d, k) = node->vec_d(d);

		for (unsigned int n = 0; n < this->N; ++n)
			coefs(k, n) = node->vec_w(n);

		node = node->next;
	}
}
