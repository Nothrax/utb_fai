#pragma once

#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cfloat>

#include <simulator/function/IFunction.hpp>
#include <simulator/structure/Point.hpp>

/**
 * Class for simulating a particle
 */
 namespace simulator::algorithm{
class Particle {
public:
	/**
	 * Constructor
	 * @param boundary one point defining symmetric boundary in which particle can move
	 * @param speedLimit maximum speed that particle can achieve
	 * @param function function for optimizing
	 */
	Particle(int numberOfSteps, int dimensions, double boundary, double speedLimit, const std::shared_ptr<function::IFunction> &function);

	/**
	 * Destructor will clear particles neighbourhood_
	 */
	~Particle();

	/**
	 * Method will initialize a random position of particle within the function boundary, calculate its fitness,
	 * check its neighbourhood_ and initialize its velocity
	 */
	void initializeState();

	/**
	 * Add particle to neighbourhood_
	 * @param particle particle to be added
	 */
	void addNeighbour(const std::shared_ptr<Particle> &particle);

	/**
	 * Make one step - calculate new position, fitness, velocity and check neighbourhood_
	 */
	void makeIteration();

	/**
	 * Position getter
	 * @return
	 */
	structure::Point getPosition();

	/**
	 * Get best fitness in particle neighbourhood_
	 * @return fitness
	 */
	double getBestNeighbourFitness();

	/**
	 * Get best position in particle neighbourhood_
	 * @return position
	 */
	structure::Point getBestNeighbourPosition();

	/**
	 * Setting best global result
	 * @param globalFitness best global fitness
	 * @param globalPoint  best global position
	 */
	void setBestGlobalResult(double globalFitness, const structure::Point& globalPoint);

	/**
	 * Setting best local result
	 * @param globalFitness best local fitness
	 * @param globalPoint  best local position
	 */
	void receiveBestNeighbourResult(const structure::Point& position, double fitness);

	/**
	 * Delete particles neighbourhood_
	 */
	void resetNeighbour();

private:
	structure::Point position_;
	structure::Point velocity_;
	double localFitness_;
	double velocityLimit_;
	double boundary_;
	std::shared_ptr<function::IFunction> function_;
	double wChange_;
	double w_ = 0.9; //todo should be changed based on steps!!
	const double c_ = 2.0;
	const double c_g_ = 2.05;
	const double c_p_ = 2.05;
	structure::Point g_i_;
	int dimensions_;

	structure::Point bestGlobalPosition_;
	structure::Point bestNeighbourPosition_;
	double bestGlobalFitness_ = DBL_MAX;
	double bestNeighbourFitness_ = DBL_MAX;
	std::vector<std::shared_ptr<Particle>> neighbourhood_;

	void calculateFitness();

	void initializePosition();

	void initializeVelocity();

	void checkVelocity();

	void checkPosition();

	void tellFitnessToNeighbours();

	void calculateNewVelocity();

	void calculateNewPosition();

	void checkFitnessWithNeighbourhood();
};
 }