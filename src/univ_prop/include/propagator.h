#ifndef UNIVERSAL_PROPAGATOR_HPP
#define UNIVERSAL_PROPAGATOR_HPP

#include <vector>
#include <string>
#include <utility> // for std::pair

class UniversalPropagator {
public:
    std::string name;
    double mu;
    double tol;

    UniversalPropagator(double mu_, const std::string& name_);
    std::pair<std::vector<double>, std::vector<double>> propagate(
        const std::vector<double>& position,
        const std::vector<double>& velocity,
        double dt
    );
};

#endif // UNIVERSAL_PROPAGATOR_HPP