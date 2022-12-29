#pragma once

#include <simulator/algorithm/Particle.hpp>
#include <simulator/function/IFunction.hpp>

#include <memory>

/**
 * Swarm of particles for simulation
 */

namespace simulator::algorithm {
struct SwarmOptions {
	int numberOfSteps{ 100 }; ///number of steps of simulation
	int numberOfParticles { 13 }; ///number of particles in simulation
	int numberOfNeighbours { 3 }; ///number of neighbours for single particle
	double boundary { 5.0 }; ///default function boundary
	int dimensionCount{ 2};
	double speedLimit { 1.0 }; ///default speed limit of particle
	double error{ 0.0001 }; ///error for accepting fitness value
	std::shared_ptr<function::IFunction> function; ///function for optimalization
};

class Swarm {
public:
	/**
	 * Constructor
	 * @param swarmOptions swarm_ options
	 */
	explicit Swarm(const SwarmOptions& swarmOptions);

	/**
	 * Destructor will reset particles
	 */
	~Swarm();

	/**
	 * Method will create particles, set neighbourhood_ and initialize particles
	 */
	void initializeSwarm();

	/**
	 * Simulate swarm_ for all steps
	 */
	void simulateSwarm();

	/**
	 * Simulate one step
	 * @return true if last step was simulated
	 */
	bool makeStep();

	/**
	 * Get best fitness
	 * @return best fitness
	 */
	double getBestFitness();

	/**
	 * Get best position
	 * @return best position
	 */
	structure::Point getBestPosition();

	/**
	 * Get actual step number
	 * @return actual step number
	 */
	int getStepNumber();

	/**
	 * Get vector of all particles position
	 * @return vector of positions
	 */
	std::vector<structure::Point> getActualPositions();

private:
	std::vector<std::shared_ptr<Particle>> swarm_;
	SwarmOptions swarmOptions_;
	double bestFitness_ = DBL_MAX;
	structure::Point bestPosition_;
	double expectedResult_;
	int stepNumber_{ 0 };

	void createSwarm();

	void initializeParticles();

	void initializeNeighbourhoods();

	void broadcastGlobalResult();

	void compareParticleWithGlobalMinimum(const std::shared_ptr<Particle> &particle);
};
}