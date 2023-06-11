#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

struct CURVE_FITING_COST {
    CURVE_FITING_COST(double x, double y) : _x (x), _y(y) {}
    template <typename T>
    bool operator() (const T* const abc, T* residual) const {
        residual[0] = T (_y) - ceres::exp(abc[0]*T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }
    const double _x = 0.0;
    const double _y = 0.0;
};

int main (int argc, char** argv) {
    double a = 1.0;
    double b = 2.0;
    double c = 1.0;
    int N = 100;
    double w_sigma = 1.0;
    cv::RNG rng;
    double abc[] = {0, 0, 0};

    std::vector<double> x_data;
    std::vector<double> y_data;

    std::cout << "generating data" << std::endl;

    for (int i = 0; i < N; ++i) {
        double x = i / 100.0;
        x_data.emplace_back(x);
        y_data.emplace_back(exp(a * x * x + b * x + c)) + rng.gaussian(w_sigma);
        std::cout << x_data[i] << " " << y_data[i] << std::endl;
    }

    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CURVE_FITING_COST, 1, 3>( 
                    new CURVE_FITING_COST(x_data[i], y_data[i])), nullptr, abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary Summary;
    //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &Summary);
    //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    std::cout << Summary.BriefReport() << std::endl;
    std::cout << "estimated a.b.c = ";
    for (auto i : abc) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return 0;
}
