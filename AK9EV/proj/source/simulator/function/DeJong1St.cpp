#include <simulator/function/DeJong1St.hpp>

#include <cmath>
namespace simulator::function{

double DeJong1st::getBoundary() {
	return boundary_;
}

double DeJong1st::getMinFitness() {
	return minValue_;
}

std::vector<structure::Point> DeJong1st::getMinPoints(int dimensionCount) {
	return {{dimensionCount, minCoordinate_}};
}

///http://www.geatbx.com/docu/fcnindex-01.html
double DeJong1st::calculateFitness(structure::Point point) {
	double fitness = 0;

	for(int dimension = 0; dimension < point.getDimensionCount(); dimension++){
		fitness += pow(point[dimension], 2);
	}

	return fitness;
}

bool DeJong1st::isWithinBoundary(structure::Point pointToCheck) {
	for(int dimension = 0; dimension < pointToCheck.getDimensionCount(); dimension++){
		if(fabs(pointToCheck[dimension]) > boundary_){
			return false;
		}
	}
	return true;
}
}