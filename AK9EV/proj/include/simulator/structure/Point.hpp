#pragma once

#include <vector>
#include <memory>

namespace simulator::structure{
class Point {
public:
	explicit Point(int dimensionCount);
	explicit Point(std::vector<double> coordinates);
	Point(int dimensionCount, double value);
	[[nodiscard]] const int getDimensionCount() const;
	double& operator[](int);
	Point& operator= (const Point& fraction);
private:
	const int dimensionCount_;
	std::vector<double> coordinates_;
};
}


