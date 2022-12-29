#include <simulator/algorithm/Particle.hpp>



namespace simulator::algorithm {
Particle::Particle(int numberOfSteps, int dimensions, double boundary, double speedLimit,
				   const std::shared_ptr<function::IFunction> &function): position_(dimensions),
																		  velocity_(dimensions),
																		  g_i_(dimensions),
																		  bestGlobalPosition_(dimensions),
																		  bestNeighbourPosition_(dimensions){
	velocityLimit_ = speedLimit;
	boundary_ = boundary;
	function_ = function;
	dimensions_ = dimensions;
	wChange_ = 0.5/numberOfSteps;
}

void Particle::initializeState() {
	initializePosition();
	calculateFitness();
	checkFitnessWithNeighbourhood();
	initializeVelocity();
	checkVelocity();
}

void Particle::initializePosition() {
	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(-boundary_, boundary_);
	for(int i = 0; i < dimensions_; i++){
		position_[i] = dist(mt);
	}

}

void Particle::initializeVelocity() {
	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(-boundary_ - localFitness_, boundary_ - localFitness_);
	for(int i = 0; i < dimensions_; i++){
		velocity_[i] = dist(mt)/2;
	}
}

void Particle::checkVelocity() {
	for(int i = 0; i < dimensions_; i++){
		if(fabs(velocity_[i]) > velocityLimit_) {
			velocity_[i] = (velocity_[i] > 0) ? velocityLimit_ : -velocityLimit_;
		}
	}
}

void Particle::calculateFitness() {
	localFitness_ = function_->calculateFitness(position_);
}

void Particle::tellFitnessToNeighbours() {
	for(auto const &particle: neighbourhood_) {
		particle->receiveBestNeighbourResult(bestNeighbourPosition_, bestNeighbourFitness_);
	}
}

void Particle::receiveBestNeighbourResult(const structure::Point& pos, double fitness) {
	if(fitness < bestNeighbourFitness_) {
		bestNeighbourFitness_ = fitness;
		bestNeighbourPosition_ = pos;
	}
}

void Particle::addNeighbour(const std::shared_ptr<Particle> &particle) {
	neighbourhood_.push_back(particle);
}

void Particle::setBestGlobalResult(double globalFitness, const structure::Point& globalPoint) {
	this->bestGlobalFitness_ = globalFitness;
	this->bestGlobalPosition_ = globalPoint;
}

structure::Point Particle::getBestNeighbourPosition() {
	return bestNeighbourPosition_;
}

double Particle::getBestNeighbourFitness() {
	return bestNeighbourFitness_;
}

void Particle::resetNeighbour() {
	neighbourhood_.clear();
}

void Particle::makeIteration() {
	calculateNewVelocity();
	calculateNewPosition();
	calculateFitness();
	checkFitnessWithNeighbourhood();
	w_ -= wChange_;
}

void Particle::calculateNewVelocity() {
	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0, c_);

	double r_p = dist(mt);
	double r_g = dist(mt);

	for(int i = 0; i < dimensions_; i++){
		velocity_[i] = w_*velocity_[i] + r_p*(bestGlobalPosition_[i] - position_[i]) + r_g*(bestNeighbourPosition_[i] - position_[i]);
	}
	checkVelocity();
}

void Particle::calculateNewPosition() {
	for(int i =0; i < dimensions_; i++){
		position_[i] += velocity_[i];
		position_[i] += velocity_[i];
	}
	checkPosition();
}

void Particle::checkPosition() {
	for(int i =0; i < dimensions_; i++){
		if(fabs(position_[i]) > boundary_) {
			position_[i] = (position_[i] > 0) ? boundary_ : -boundary_;
			velocity_[i] = 0;
		}
	}
}

void Particle::checkFitnessWithNeighbourhood() {
	if(localFitness_ < bestNeighbourFitness_) {
		bestNeighbourFitness_ = localFitness_;
		bestNeighbourPosition_ = position_;
		tellFitnessToNeighbours();
	}
}

structure::Point Particle::getPosition() {
	return position_;
}

Particle::~Particle() {
	neighbourhood_.clear();
}
}
