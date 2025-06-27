#include <iostream>
#include <cmath>

double f(double chi, double r0, double C){
    return 1 - (chi * chi) / (r0) - C;
}

double d_f(double chi, double r0, double r, double S, double z, double mu) {
    return ((std::sqrt(mu)) / (r * r0)) * chi * (z * S - 1);
}

double g(double chi, double mu, double S, double dt){
    return dt - (1 / (std::sqrt(mu))) * chi * chi * chi * S;
}

double d_g(double chi, double r, double C){
    return 1 - ((chi * chi) / r) * C;
}