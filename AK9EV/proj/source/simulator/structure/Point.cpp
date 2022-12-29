#include <simulator/structure/Point.hpp>
#include <utility>
#include <stdexcept>

namespace simulator::structure{

Point::Point(std::vector<double> coordinates):dimensionCount_(coordinates.size()) {
	coordinates_ = std::move(coordinates);
}

const int Point::getDimensionCount() const {
	return dimensionCount_;
}

double& Point::operator[](int index) {
	if(index >= dimensionCount_ ){
		throw std::out_of_range("Point subscript is out of range.");
	}
	return coordinates_.at(index);
}

Point::Point(int dimensionCount, double value):dimensionCount_(dimensionCount) {
	coordinates_.reserve(dimensionCount);
	for(int i = 0; i < dimensionCount; i++) {
		coordinates_.push_back(value);
	}
}

Point::Point(int dimensionCount):dimensionCount_(dimensionCount) {
	coordinates_.reserve(dimensionCount);
	for(int i = 0; i < dimensionCount; i++) {
		coordinates_.push_back(0);
	}
}

Point &Point::operator=(const Point &point) {
	if(dimensionCount_ != point.dimensionCount_){
		throw std::out_of_range{"Trying to assign point of different dimensions"};
	}
	for(int dimension = 0; dimension < dimensionCount_; dimension++){
		coordinates_[dimension] = point.coordinates_[dimension];
	}
	return *this;
}


}
