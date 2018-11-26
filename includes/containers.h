#pragma once

#include<string>

#include "Eigen/Dense"

#include "typedef.h"

namespace sva {

	struct Node {

		Node(int D, int N);
		Node(int D, int N, const Eigen::VectorXd &dic);
		Node(int D, int N, const Eigen::VectorXd &dic, const Eigen::VectorXd &coefs);

			// A Dx1 vector
		Eigen::VectorXd vec_d; 

			// A 1xN vector
		Eigen::VectorXd vec_w;

		unsigned int mk = 0;

			// Next node
		Node *next = nullptr;
	};


	class DictAndCoef {

	public:
		/**
		 * @brief constructor
		 * @details [long description]
		 * 
		 * @param D dimension of Data
		 * @param N number of observations
		 */
		DictAndCoef(int D, int N);

		~DictAndCoef();

		void save(std::string filename);

		/**
		 * @brief Sum up dictionary
		 * @details [long description]
		 */
		void print();

		/**
		 * @brief Print the first k column of D
		 * @details [long description]
		 * 
		 * @param k [description]
		 */
		void print_KcolOfD(int k);

		void add(const Eigen::VectorXd& new_d);
		void add(const Eigen::VectorXd& new_d, int n, double w);
		void add(const Eigen::VectorXd& new_d, const Eigen::VectorXd& new_coefs);

		/**
		 * @brief clean node where m_k == 0
		 * @details [long description]
		 */
		void clean_unuse_node();

		void setCoefToZero(int n);

		void setCoef(const Eigen::VectorXd &coefs, int n);
		
		inline int get_K() const { return this->length; }
		inline int get_D() const { return this->D; }
		inline int get_N() const { return this->N; }

		unsigned int get_mk(unsigned int k) const;
		void get_support(int n, Eigen::VectorXi &output) const;

		void computeDWn(int n, Eigen::VectorXd &output) const;

		void computeDHWn(int n, const Eigen::custom::VectorXb& cvecMask, Eigen::VectorXd &vecOutput) const;

		void chainToMatrix(Eigen::MatrixXd &dic, Eigen::MatrixXd &coefs) const;

		void get_featk(int k, Eigen::MatrixXd &dic, int t) const;

		sva::Node *get_head() { return this->head; }

		const sva::Node *get_head() const { return this->head; }

	protected:

		unsigned int D, N;
		unsigned int length;
		sva::Node *head;
		sva::Node *last;
	};
}