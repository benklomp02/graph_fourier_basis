#pragma once
#include <Eigen/Dense>
#include <vector>
#include "generator.hpp"

generator<Eigen::MatrixXd> get_all_partition_matrices(int n, int m);