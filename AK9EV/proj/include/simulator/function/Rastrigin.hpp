#pragma once

#include <simulator/function/IFunction.hpp>

namespace simulator::function {
    class Rastrigin: public IFunction{
    public:
        double calculateFitness(structure::Point point) override;

        std::vector<structure::Point> getMinPoints(int dimensionCount) override;

        double getMinFitness() override;

        double getBoundary() override;

        bool isWithinBoundary(structure::Point pointToCheck) override;

    private:
        const double minCoordinate_ = 0;
        const double minValue_ = 0;
        const double boundary_ = 5.12;
    };
}