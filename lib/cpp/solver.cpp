#include "solver.hpp"

Eigen::VectorXd solve_minimisation_problem(const Eigen::MatrixXd &M, const Eigen::MatrixXd *U = nullptr, bool is_constant = false)
{
    Eigen::VectorXd _x;
    if (is_constant)
    {
        double c1 = M.col(0).sum();
        double c2 = M.col(1).sum();
        Eigen::Vector2d a(1.0, -c1 / c2);
        _x = M * a;
    }
    else
    {
        Eigen::MatrixXd X = U->transpose() * M;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeFullV);
        Eigen::VectorXd nullVec = svd.matrixV().col(X.cols() - 1);
        _x = M * nullVec;
    }
    return _x / _x.norm();
}