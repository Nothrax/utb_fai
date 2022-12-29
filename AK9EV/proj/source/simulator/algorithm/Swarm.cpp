#include <simulator/algorithm/Swarm.hpp>

namespace simulator::algorithm{
Swarm::Swarm(const SwarmOptions& swarmOptions): bestPosition_(swarmOptions.dimensionCount) {
	swarmOptions_ = swarmOptions;
}

void Swarm::initializeSwarm() {
	createSwarm();
	initializeNeighbourhoods();
	initializeParticles();

	expectedResult_ = swarmOptions_.function->getMinFitness();
}

void Swarm::createSwarm() {
	for (int i = 0; i < swarmOptions_.numberOfParticles; i++) {
		swarm_.push_back(
				std::make_shared<Particle>(swarmOptions_.numberOfSteps,swarmOptions_.dimensionCount, swarmOptions_.boundary, swarmOptions_.speedLimit, swarmOptions_.function));
	}
}

void Swarm::initializeParticles() {
	for (auto const &particle: swarm_) {
		particle->initializeState();
		compareParticleWithGlobalMinimum(particle);
	}
}

structure::Point Swarm::getBestPosition() {
	return bestPosition_;
}

double Swarm::getBestFitness() {
	return bestFitness_;
}

void Swarm::initializeNeighbourhoods() {
	for (int i = 0; i < swarmOptions_.numberOfParticles; i++) {
		int index = (i - 1);
		if (index < 0) {
			index = swarmOptions_.numberOfParticles - 1;
		}
		swarm_.at(i)->addNeighbour(swarm_.at(index));
		swarm_.at(i)->addNeighbour(swarm_.at(i));
		index = (i + 1)%swarmOptions_.numberOfParticles;
		swarm_.at(i)->addNeighbour(swarm_.at(index));
	}
}

void Swarm::broadcastGlobalResult() {
	for (auto const &particle: swarm_) {
		particle->setBestGlobalResult(bestFitness_, bestPosition_);
	}
}

void Swarm::simulateSwarm() {
	bool end = false;
	while (!end) {
		end = makeStep();
	}
}

bool Swarm::makeStep() {
	for (auto const &particle: swarm_) {
		particle->makeIteration();
		compareParticleWithGlobalMinimum(particle);
	}


	///fabs(expectedResult_ - bestFitness_) < swarmOptions_.error ||
	if (stepNumber_ == swarmOptions_.numberOfSteps) {
		return true;
	}
	stepNumber_++;
	return false;
}

void Swarm::compareParticleWithGlobalMinimum(const std::shared_ptr<Particle> &particle) {
	double particleFitness = particle->getBestNeighbourFitness();
	if (particleFitness < bestFitness_) {
		bestFitness_ = particleFitness;
		bestPosition_ = particle->getBestNeighbourPosition();
		broadcastGlobalResult();
	}
}

std::vector<structure::Point> Swarm::getActualPositions() {
	std::vector<structure::Point> positions;
	for (auto const &particle: swarm_) {
		positions.push_back(particle->getPosition());
	}
	return positions;
}

int Swarm::getStepNumber() {
	return stepNumber_;
}

Swarm::~Swarm() {
	for (const auto &particle: swarm_) {
		particle->resetNeighbour();
	}
	swarm_.clear();
}
}