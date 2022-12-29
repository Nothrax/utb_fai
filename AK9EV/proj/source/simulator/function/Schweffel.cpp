#include <simulator/function/Schweffel.hpp>

#include <cmath>

namespace simulator::function {

double Schweffel::getBoundary() {
	return boundary_;
}

double Schweffel::getMinFitness() {
	return minValue_;
}

std::vector<structure::Point> Schweffel::getMinPoints(int dimensionCount) {
	return {{dimensionCount, minCoordinate_}};
}

///https://www.sfu.ca/~ssurjano/schwef.html
double Schweffel::calculateFitness(structure::Point point) {
	double fitness;

	fitness = 418.9829*point.getDimensionCount();

	for(int dimension = 0; dimension < point.getDimensionCount(); dimension++){
		fitness -= point[dimension]*sin(sqrt(fabs(point[dimension])));
	}

	return fitness;
}

bool Schweffel::isWithinBoundary(structure::Point pointToCheck) {
	for(int dimension = 0; dimension < pointToCheck.getDimensionCount(); dimension++){
		if(fabs(pointToCheck[dimension]) > boundary_){
			return false;
		}
	}
	return true;
}
}