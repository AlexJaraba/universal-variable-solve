#pragma once
#include <functional>

double Newton_Solver(std::function<double(double)> function,
                      std::function<double(double)> d_function,
                      double x0,
                      double tolerance,
                      int maxIterations);