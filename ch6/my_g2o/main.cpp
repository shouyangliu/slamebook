#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double* update) {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(std::istream& in) {}
    virtual bool write(std::ostream& out) const {}
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
    void computeError() {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }
    virtual bool read(std::istream& in) {}
    virtual bool write(std::ostream& out) const {}

    double _x;
};

int main(int argc, char** argv) {
    double a = 1.0;
    double b = 2.0;
    double c = 1.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abc[3] = {0, 0, 0};

    std::vector<double> x_data;
    std::vector<double> y_data;
    std::cout << "generating data" << std::endl;
    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.emplace_back(x);
        y_data.emplace_back(exp(a * x * x + b * x + c) + rng.gaussian(w_sigma));
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverDense<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));
    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    //Block* solver_ptr = new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgrothimAlogrithmGaussNewton(solver_ptr);
    //
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0));
    v->setId(0);
    optimizer.addVertex(v);

    for (int i = 0; i < N; ++i) {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma));
        optimizer.addEdge(edge);
    }
    std::cout << "start optimization" <<std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    std::cout << "solve the problem" << std::endl;
    Eigen::Vector3d abc_estimate = v->estimate();
    std::cout << "estimated model: " << abc_estimate.transpose() << std::endl;
    return 0;
}
