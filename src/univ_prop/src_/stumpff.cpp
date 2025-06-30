#include <cmath>

// Stumpff function C(z)
double stumpff_C(double z) {
    if (z > 0) {
        double sz = std::sqrt(z);
        return (1.0 - std::cos(sz)) / z;
    } else if (z < 0) {
        double sz = std::sqrt(-z);
        return (std::cosh(sz) - 1.0) / (-z);
    } else {
        return 0.5;
    }
}

// Stumpff function S(z)
double stumpff_S(double z) {
    if (z > 0) {
        double sz = std::sqrt(z);
        return (sz - std::sin(sz)) / (sz * sz * sz);
    } else if (z < 0) {
        double sz = std::sqrt(-z);
        return (std::sinh(sz) - sz) / (sz * sz * sz);
    } else {
        return 1.0 / 6.0;
    }
}