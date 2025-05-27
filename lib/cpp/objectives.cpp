#include "objectives.hpp"

// The objective function working for both directed and undirected graphs.
double S(const Eigen::VectorXd &x, const Eigen::MatrixXd &weights)
{
    int n = x.size();
    double sum = 0.;
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
        {
            if (x[i] > x[j])
                sum += (x[i] - x[j]) * weights(i, j);
            else
                sum += (x[j] - x[i]) * weights(j, i);
        }
    return sum;
}