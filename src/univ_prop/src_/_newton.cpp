#include <cmath>
#include <limits>
#include <iostream>
#include <functional>

// Now takes double-valued functions and variables
double Newton_Solver(std::function<double(double)> function,
                     std::function<double(double)> d_function,
                     double x0,
                     double tolerance,
                     int maxIterations) {
    double x = x0;

    for (int i = 0; i < maxIterations; ++i) {
        double f_x = function(x);
        double df_x = d_function(x);

        if (df_x == 0) {
            std::cerr << "Derivative is zero, cannot proceed.\n";
            return std::numeric_limits<double>::quiet_NaN();
        }

        double x_new = x - f_x / df_x;

        if (std::abs(x_new - x) < tolerance) {
            return x_new;
        }

        x = x_new;
    }

    std::cerr << "Max iterations reached without convergence.\n";
    return std::numeric_limits<double>::quiet_NaN();
}