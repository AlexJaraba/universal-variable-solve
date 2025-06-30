#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <limits>

#include "coef.h"
#include "stumpff.h"
#include "_newton.h"

// Compute Euclidean norm of vector
double norm(const std::vector<double>& v) {
    double sum_sq = 0.0;
    for (double x : v) sum_sq += x * x;
    return std::sqrt(sum_sq);
}

// solve_chi function
double solve_chi(double mu, double alpha, const std::vector<double>& r0, double vr, double dt,
                 double tol=1e-8, int max_iter=100) 
{
    double chi0 = std::sqrt(mu) * dt * std::abs(alpha);
    double r = norm(r0);

    if (!std::isfinite(chi0)) {
        throw std::runtime_error("Initial guess chi0 is not finite");
    }
    if (std::abs(alpha * chi0 * chi0) > 1e6) {
        throw std::runtime_error("Chi0 guess leads to extreme z value");
    }

    // Define F(chi)
    auto F = [&](double chi) {
        double z = alpha * chi * chi;
        double C = stumpff_C(z);
        double S = stumpff_S(z);
        return (r * vr / std::sqrt(mu)) * chi * chi * C 
             + (1 - r * alpha) * chi * chi * chi * S 
             + r * chi 
             - std::sqrt(mu) * dt;
    };

    // Define derivative dF(chi)
    auto dF = [&](double chi) {
        double z = alpha * chi * chi;
        double C = stumpff_C(z);
        double S = stumpff_S(z);
        return (r * vr / std::sqrt(mu)) * chi * (1 - z * S) 
             + (1 - r * alpha) * chi * chi * C 
             + r;
    };

    return Newton_Solver(F, dF, chi0, tol, max_iter);
}