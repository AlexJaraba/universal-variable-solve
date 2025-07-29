#include "kepler.h"
#include "stumpff.h"
#include "coef.h"
#include "propagator.h"

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
            double time_step) 
        {
            const std::vector<double>& r0_vec = Position;
            const std::vector<double>& v0_vec = Velocity;

            double r0 = norm(Position);
            double v0 = norm(Velocity);
            double vr = (r0_vec[0] * r0_vec[0] + r0_vec[1] * v0_vec[1] + v0_vec[2] * v0_vec[2]) / r0;

            double alpha = 2.0 / (r0) - (v0 * v0) / mu;

            double chi;
            try {
                chi = solve_chi(mu, alpha, Position, vr, time_step, tol);
            } catch (const std::runtime_error& e){
                std::cerr << "Propagation failed: " << e.what() << std::endl;
                throw;
            }

            double z = alpha * chi * chi;
            double C = stumpff_C(z);
            double S = stumpff_S(z);

            double _f = f(chi, r0, C);
            double _g = g(chi, mu, S, time_step);

            std::vector<double> r_vec(3);
            for (int i =0; i < 3; ++i) {
                r_vec[i] = r0_vec[i] * _f + v0_vec[i] * _g;
            }
            double r = norm(r_vec);

            double _d_f = d_f(chi, r0, r, S, z, mu);
            double _d_g = d_g(chi, r, C);

            std::vector<double> v_vec(3);
            for (int i = 0; i < 3; ++i) {
                v_vec[i] = r0_vec[i] * _d_f + v0_vec[i] * _d_g;
            }

            return {r_vec, v_vec};
        }
    };
}