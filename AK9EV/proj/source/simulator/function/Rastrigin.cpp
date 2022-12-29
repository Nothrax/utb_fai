#include <simulator/function/Rastrigin.hpp>

#include <cmath>



namespace simulator::function {

double Rastrigin::getBoundary() {
	return boundary_;
}

double Rastrigin::getMinFitness() {
	return minValue_;
}

std::vector<structure::Point> Rastrigin::getMinPoints(int dimensionCount) {
	return {{ dimensionCount, minCoordinate_ }};
}

///http://www.sfu.ca/~ssurjano/rastr.html
double Rastrigin::calculateFitness(structure::Point point) {
	double fitness = 10*point.getDimensionCount();

	for(int dimension = 0; dimension < point.getDimensionCount(); dimension++) {
		fitness += pow(point[dimension], 2) - 10*cos(2*M_PI*point[dimension]);
	}

	return fitness;
}

bool Rastrigin::isWithinBoundary(structure::Point pointToCheck) {
	for(int dimension = 0; dimension < pointToCheck.getDimensionCount(); dimension++) {
		if(fabs(pointToCheck[dimension]) > boundary_) {
			return false;
		}
	}
	return true;
}
}