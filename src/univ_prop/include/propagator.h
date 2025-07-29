#ifndef UNIV_PROP_H
#define UNIV_PROP_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace univ_prop {
    class UniversalPropagator {
    public:
        std::string name;
        double mu;
        double tol;

        UniversalPropagator(double mu_, const std::string& name_)
            : name(name_), mu(mu_), tol(1e-8) {}

        std::pair<std::vector<double>, std::vector<double>> propagate(
            const std::vector<double>& Position, 
            const std::vector<double>& Velocity, 
            double time_step);
    };
}

#endif