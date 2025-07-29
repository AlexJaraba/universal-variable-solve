#pragma once
#include <functional>

double f(double chi, double r0, double C);
double d_f(double chi, double r0, double r, double S, double z, double mu);
double g(double chi, double mu, double S, double dt);
double d_g(double chi, double r, double C);