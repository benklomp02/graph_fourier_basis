#pragma once
#include <Eigen/Dense>
#include <vector>

Eigen::VectorXd solve_minimisation_problem(const Eigen::MatrixXd &M, const Eigen::MatrixXd *U, bool is_constant);