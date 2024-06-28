#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

class Optimizer {
  public:
    virtual void update(double** weights, double** gradWeights, double* bias, double* gradBias, double lr) = 0;
    virtual ~Optimizer() = default;
};

class Adam: public Optimizer {
  private:
    double beta1 = 0.9;
    double beta1PowT = 1.0;
    double beta2 = 0.999;
    double beta2PowT = 1.0;
    double epsilon = 1e-8;
    int row;
    int column;
    double** m;
    double** v;
  public:
    Adam(int row, int column);
    void update(double** weights, double** gradWeights, double* bias, double* gradBias, double lr) final;
    ~Adam() final;
};

class RMSprop: public Optimizer {
  private:
    double rho = 0.9;
    double epsilon = 1e-8;
    int row;
    int column;
    double** v;
  public:
    RMSprop(int row, int column);
    void update(double** weights, double** gradWeights, double* bias, double* gradBias, double lr) final;
    ~RMSprop() final;
};
#endif //OPTIMIZERS_H