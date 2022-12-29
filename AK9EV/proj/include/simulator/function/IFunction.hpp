#pragma once

#include <simulator/structure/Point.hpp>



namespace simulator::function {

enum class FunctionEnum{
	SCHWEFFEL = 0,
	DEJONG_1,
	DEJONG_2,
    RASTRIGIN
};
class IFunction {
public:

	virtual double calculateFitness(structure::Point point) = 0;

	virtual std::vector<structure::Point> getMinPoints(int dimensionCount) = 0;

	virtual double getMinFitness() = 0;

	virtual double getBoundary() = 0;

	virtual bool isWithinBoundary(structure::Point pointToCheck) = 0;

protected:
	const double _pi { 3.14159265358979323846 };
	const double _e { 2.71828182845904523536 };

};
}

