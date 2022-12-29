#include <simulator/function/DeJong2Nd.hpp>

#include <cmath>



namespace simulator::function {

double DeJong2Nd::getBoundary() {
	return boundary_;
}

double DeJong2Nd::getMinFitness() {
	return minValue_;
}

std::vector<structure::Point> DeJong2Nd::getMinPoints(int dimensionCount) {
	return { {dimensionCount, minCoordinate_} };
}

///http://www.geatbx.com/docu/fcnindex-01.html
double DeJong2Nd::calculateFitness(structure::Point point) {
	double fitness = 0;

	for(int dimension = 0; dimension < point.getDimensionCount() - 1; dimension++) {
		fitness += 100*pow((point[dimension + 1] - pow(point[dimension], 2)),2) + pow((1 - point[dimension]), 2);
	}

	return fitness;
}

bool DeJong2Nd::isWithinBoundary(structure::Point pointToCheck) {
	for(int dimension = 0; dimension < pointToCheck.getDimensionCount(); dimension++) {
		if(fabs(pointToCheck[dimension]) > boundary_) {
			return false;
		}
	}
	return true;
}
}
