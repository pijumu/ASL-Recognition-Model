#include "Optimizers.h"
#include <cstring>
#include <cmath>

Adam::Adam(int row, int column): row(row+1), column(column), m(new double* [row]), v(new double* [row]) {
    for (int i=0; i<row+1; ++i) {
        m[i] = new double [column];
        v[i] = new double [column];
        memset(m[i], 0.0, column);
        memset(v[i], 0.0, column);
    }
}

void Adam::update(double** weights, double** gradWeights, double* bias, double* gradBias, double lr) {
    beta1PowT *= beta1;
    beta2PowT *= beta2;
    for (int i=0; i<row-1; ++i) {
        for (int j=0; j<column; ++j) {
            m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradWeights[i][j];
            v[i][j] = beta2 * v[i][j] + (1 - beta2) * gradWeights[i][j] * gradWeights[i][j];
            double m_ = m[i][j] / (1.0 - beta1PowT);
            double v_ = v[i][j] / (1.0 - beta1PowT);
            weights[i][j] -= lr * m_ / (epsilon + std::sqrt(v_));
        }
    }
    for (int j=0; j<column; ++j) {
        m[row-1][j] = beta1 * m[row-1][j] + (1 - beta1) * gradBias[j];
        v[row-1][j] = beta2 * v[row-1][j] + (1 - beta2) * gradBias[j] * gradBias[j];
        double m_ = m[row-1][j] / (1.0 - beta1PowT);
        double v_ = v[row-1][j] / (1.0 - beta1PowT);
        bias[j] -= lr * m_ / (epsilon + std::sqrt(v_));
    }
}

Adam::~Adam() {
    for (int i=0; i<row; ++i) {
        delete[] m[i];
        delete[] v[i];
    }
    delete[] m;
    delete[] v;
}

RMSprop::RMSprop(int row, int column): row(row+1), column(column), v(new double* [row]) {
    for (int i=0; i<row+1; ++i) {
        v[i] = new double [column];
        memset(v[i], 0.0, column);
    }
}

void RMSprop::update(double** weights, double** gradWeights, double* bias, double* gradBias, double lr) {
    for (int i=0; i<row-1; ++i) {
        for (int j=0; j<column; ++j) {
            v[i][j] = rho * v[i][j] + (1 - rho) * gradWeights[i][j] * gradWeights[i][j];
            weights[i][j] -= lr * gradWeights[i][j] / (epsilon + std::sqrt(v[i][j]));
        }
    }
    for (int j=0; j<column; ++j) {
        v[row-1][j] = rho * v[row-1][j] + (1 - rho) * gradBias[j] * gradBias[j];
        bias[j] -= lr * gradBias[j] / (epsilon + std::sqrt(v[row-1][j]));
    }
}

RMSprop::~RMSprop() {
    for (int i=0; i<row; ++i) {
        delete[] v[i];
    }
    delete[] v;
}