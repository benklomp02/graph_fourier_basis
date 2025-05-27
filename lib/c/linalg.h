#ifndef LINALG_H
#define LINALG_H

typedef double (*objective_fn)(int n, const int *card, const double *memo, int i, int j);

int objective(int n, const int *card, const double *memo, int i, int j);

int objective2(int n, const int *card, const double *memo, int i, int j);

int objective3(int n, const int *card, const double *memo, int i, int j);

int objective4(int n, const int *card, const double *memo, int i, int j);

int objective_max(int n, const int *card, const double *memo, int i, int j);

int objective_min(int n, const int *card, const double *memo, int i, int j);

int objective_harmonic_mean(int n, const int *card, const double *memo, int i, int j);

int objective_sym(int n, const int *card, const double *memo, int i, int j);

static int byte_array_count(const unsigned char *arr, int n);

int *arg_max_greedy(int n_clusters, int n_vertices, const unsigned char *tau, const double *memo, objective_fn obj_fn);

#endif