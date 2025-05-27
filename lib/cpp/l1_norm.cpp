#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <limits>

#include "objectives.hpp"
#include "partition_matrix.hpp"
#include "solver.hpp"

void _expand_basis_set(std::vector<Eigen::VectorXd> &basis, int k, const Eigen::MatrixXd &weights, int n)
{
    Eigen::MatrixXd U(n, basis.size());
    for (size_t i = 0; i < basis.size(); ++i)
        U.col(i) = basis[i];

    auto partition_matrices = get_all_partition_matrices(n, k);
    Eigen::VectorXd best_u;
    double best_score = std::numeric_limits<double>::infinity();

    for (const auto &M : partition_matrices)
    {
        Eigen::VectorXd uk = solve_minimisation_problem(M, &U, false);
        double score = S(uk, weights);
        if (score < best_score)
        {
            best_score = score;
            best_u = uk;
        }
    }
    basis.push_back(best_u);
}

Eigen::MatrixXd compute_l1_norm_basis(int n, const Eigen::MatrixXd &weights)
{
    Eigen::VectorXd u1 = Eigen::VectorXd::Ones(n) / sqrt(n);

    auto partition_matrices = get_all_partition_matrices(n, 2);
    Eigen::VectorXd u2;
    double best_score = std::numeric_limits<double>::infinity();

    for (const Eigen::MatrixXd &M : partition_matrices)
    {
        Eigen::VectorXd x = solve_minimisation_problem(M, nullptr, true);
        double score = S(x, weights);
        if (score < best_score)
        {
            best_score = score;
            u2 = x;
        }
    }
    std::vector<Eigen::VectorXd> basis = {u1, u2};
    for (int k = 3; k <= n; ++k)
        _expand_basis_set(basis, k, weights, n);

    Eigen::MatrixXd result(n, basis.size());
    for (size_t i = 0; i < basis.size(); ++i)
        result.col(i) = basis[i];

    return result;
}

extern "C"
{
    double *compute_l1_norm_basis_c(int n, const double *weights_array, int *out_rows, int *out_cols)
    {
        try
        {
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weights(weights_array, n, n);
            Eigen::MatrixXd basis = compute_l1_norm_basis(n, weights);

            *out_rows = basis.rows();
            *out_cols = basis.cols();

            double *result = (double *)malloc(sizeof(double) * (*out_rows) * (*out_cols));
            if (!result)
                throw std::runtime_error("Memory allocation failed");

            for (int i = 0; i < *out_rows; ++i)
                for (int j = 0; j < *out_cols; ++j)
                    result[i + j * (*out_rows)] = basis(i, j);

            return result;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void free_allocated_array(double *ptr)
    {
        free(ptr);
    }
}