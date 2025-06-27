#pragma once
#include <functional>

double solve_chi(double mu, double alpha, const std::vector<double>& r0, double vr, double dt,
                  double tol=1e-8, int max_iter=100);