#include <stdlib.h>

typedef double (*objective_fn)(int n, const int *card, const double *memo, int i, int j);

// Objective functions for the greedy algorithm.
// --------------------------------------------------------------------------------------
int objective(int n, const int *card, const double *memo, int i, int j)
{
    double denom = (double)card[i] * (double)card[j];
    return memo[i * n + j] / denom;
}
// Objective functions for the greedy algorithm using different denominators.
// --------------------------------------------------------------------------------------

// sum
int objective2(int n, const int *card, const double *memo, int i, int j)
{
    double denom = (double)(card[i] + card[j]);
    return memo[i * n + j] / denom;
}

// max
int objective3(int n, const int *card, const double *memo, int i, int j)
{
    double denom = (double)(card[i] > card[j] ? card[i] : card[j]);
    return memo[i * n + j] / denom;
}

// min
int objective4(int n, const int *card, const double *memo, int i, int j)
{
    double denom = (double)(card[i] < card[j] ? card[i] : card[j]);
    return memo[i * n + j] / denom;
}

// Alternative objective functions for the greedy algorithm in the directed case.
// --------------------------------------------------------------------------------------

int objective_max(int n, const int *card, const double *memo, int i, int j)
{
    double denom = (double)card[i] * (double)card[j];
    double Wij = memo[i * n + j];
    double Wji = memo[j * n + i];
    return (Wij > Wji ? Wij : Wji) / denom;
}

int objective_min(int n, const int *card, const double *memo, int i, int j)
{
    double denom = (double)card[i] * (double)card[j];
    double Wij = memo[i * n + j];
    double Wji = memo[j * n + i];
    return (Wij < Wji ? Wij : Wji) / denom;
}

int objective_sym(int n, const int *card, const double *memo, int i, int j)
{
    double Wij = memo[i * n + j];
    double Wji = memo[j * n + i];
    return (Wij + Wji) / 2;
}

int objective_harmonic_mean(int n, const int *card, const double *memo, int i, int j)
{
    double Wij = memo[i * n + j];
    double Wji = memo[j * n + i];
    double denom = Wij + Wji;
    return 2.0 * Wij * Wji / denom;
}

// Objective function to approximate the laplacian cost. (Note: these are not used in the greedy algorithm.)
// --------------------------------------------------------------------------------------

int objective_laplacian_by_majority(int n, const int *card, const double *memo, int i, int j)
{
    return -1;
}

int objective_laplacian_by_median(int n, const int *card, const double *memo, int i, int j)
{
    return -1;
}

int objective_laplacian_by_mean(int n, const int *card, const double *memo, int i, int j)
{
    return -1;
}

// Argmax algorithm with generic objective function.
// --------------------------------------------------------------------------------------

// Count the number of 1s in a length‐n byte array.
static int byte_array_count(const unsigned char *arr, int n)
{
    int cnt = 0;
    for (int i = 0; i < n; i++)
        cnt += arr[i];
    return cnt;
}

int *arg_max_greedy(int n_clusters,
                    int n_vertices,
                    const unsigned char *tau,
                    const double *memo,
                    objective_fn obj_fn)
{
    int *result = malloc(2 * sizeof(int));
    int *card = malloc(n_clusters * sizeof(int));
    for (int i = 0; i < n_clusters; i++)
        card[i] = byte_array_count(tau + i * (size_t)n_vertices, n_vertices);
    double best_val = -1;
    int best_i = -1, best_j = -1;
    for (int i = 1; i < n_clusters; i++)
    {
        for (int j = 0; j < i; j++)
        {
            double alt;
            alt = obj_fn(n_clusters, card, memo, i, j);
            if (alt > best_val)
            {
                best_val = alt;
                best_i = i;
                best_j = j;
            }
        }
    }
    free(card);
    result[0] = best_i;
    result[1] = best_j;
    return result;
}