
#include <simulator/Simulator.hpp>
#include <simulator/function/FunctionFactory.hpp>
#include <simulator/algorithm/Swarm.hpp>

#include <iostream>
#include <random>



namespace simulator {

void Simulator::simulate(const std::string &path, int numberOfRuns, int dimensionCount, int fes,
						 function::FunctionEnum functionType, AlgorithmType algoType) {
	function_ = function::FunctionFactory::makeFunction(functionType);
	fes_ = fes;
	dimensionCount_ = dimensionCount;
	outputFile_.open(path + ".csv");
	numberOfRuns_ = numberOfRuns;
	writeHeader();



	//todo algorithm should be class on its own
	for(actualRun_ = 0; actualRun_ < numberOfRuns_; actualRun_++) {
		Run run;
		run.runNumber = actualRun_;
		results_.push_back(run);
		switch(algoType) {
			case AlgorithmType::RANDOM_SEARCH:
				randomSearch();
				break;
			case AlgorithmType::HILL_CLIMBING:
				hillClimbing();
				break;
			case AlgorithmType::PSU:
				psu();
				break;
		}
	}

	writeRuns();
	results_.clear();

	outputFile_.close();
}

void Simulator::psu() {
	algorithm::SwarmOptions swarmOptions;
	swarmOptions.dimensionCount = dimensionCount_;
	swarmOptions.numberOfParticles = 50;
	numberOfSteps_ = fes_/swarmOptions.numberOfParticles;
	swarmOptions.numberOfSteps = numberOfSteps_;
	swarmOptions.boundary = function_->getBoundary();
	swarmOptions.speedLimit=(function_->getBoundary()*2)/50.0;
	swarmOptions.error = 0.0;
	swarmOptions.function = function_;


	algorithm::Swarm swarm(swarmOptions);
	swarm.initializeSwarm();

	while(!swarm.makeStep()) {
		auto bestPosition = swarm.getBestPosition();
		auto bestFitness = swarm.getBestFitness();
		results_.at(actualRun_).costFunctionValues.push_back(bestFitness);
	}

}

void Simulator::randomSearch() {
	structure::Point bestPosition { dimensionCount_ };
	structure::Point actualPosition(dimensionCount_);
	double bestFitness, actualFitness;
	numberOfSteps_ = fes_;

	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_real_distribution<double> gen(-function_->getBoundary(), function_->getBoundary());

	for(int i = 0; i < dimensionCount_; i++) {
		actualPosition[i] = gen(mt);
	}

	bestPosition = actualPosition;
	bestFitness = function_->calculateFitness(bestPosition);

	for(int step = 0; step < numberOfSteps_; step++) {
		///generate new point
		for(int i = 0; i < dimensionCount_; i++) {
			actualPosition[i] = gen(mt);
		}

		if(!function_->isWithinBoundary(actualPosition)) {
			throw std::out_of_range("Somehow I ended up outside of the range");
		}

		actualFitness = function_->calculateFitness(actualPosition);
		if(actualFitness < bestFitness) {
			bestPosition = actualPosition;
			bestFitness = actualFitness;
		}
		results_.at(actualRun_).costFunctionValues.push_back(bestFitness);
	}

}

void Simulator::hillClimbing() {
	double maxStep = function_->getBoundary()/10; /// max 10 % step allowed
	structure::Point bestPosition { dimensionCount_ };
	double bestFitness, actualFitness;
	int neighbourhoodSize = 10;
	numberOfSteps_ = fes_/neighbourhoodSize;

	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_real_distribution<double> pointGen(-function_->getBoundary(), function_->getBoundary());
	std::uniform_real_distribution<double> stepGen(-function_->getBoundary()/10, function_->getBoundary()/10);

	for(int i = 0; i < dimensionCount_; i++) {
		bestPosition[i] = pointGen(mt);
	}

	bestFitness = function_->calculateFitness(bestPosition);


	for(int step = 0; step < numberOfSteps_; step++) {
		//todo check in bounds
		std::vector<structure::Point> bestNeighbourhoodPositions(
				{}); /// stochastic hill climb, we chose random better solution

		for(int neigbour = 0; neigbour < neighbourhoodSize; neigbour++) {
			structure::Point newNeighbourPosition(dimensionCount_);
			newNeighbourPosition = bestPosition;
			for(int dimension = 0; dimension < dimensionCount_; dimension++) {
				double stepOffset = stepGen(mt);
				newNeighbourPosition[dimension] += stepOffset;
				///trim if outside of boundary
				if(newNeighbourPosition[dimension] > function_->getBoundary()) {
					newNeighbourPosition[dimension] = function_->getBoundary();
				} else if(newNeighbourPosition[dimension] < -function_->getBoundary()) {
					newNeighbourPosition[dimension] = -function_->getBoundary();
				}
			}

			double newNeighbourFitness = function_->calculateFitness(newNeighbourPosition);
			if(newNeighbourFitness < bestFitness) {
				bestNeighbourhoodPositions.push_back(newNeighbourPosition);
			}
		}
		if(!bestNeighbourhoodPositions.empty()) {
			std::uniform_int_distribution<int> indexGen(0, bestNeighbourhoodPositions.size() - 1);
			auto index = indexGen(mt);


			bestFitness = function_->calculateFitness(bestNeighbourhoodPositions.at(index));
			bestPosition = bestNeighbourhoodPositions.at(index);
		}
		results_.at(actualRun_).costFunctionValues.push_back(bestFitness);
	}
}

Simulator::~Simulator() {
	if(outputFile_.is_open()) {
		outputFile_.close();
	}
}

void Simulator::writeRuns() {
	for(int i = 0; i < numberOfSteps_; i++) {
		double average = 0;
		for(int j = 0; j < numberOfRuns_; j++) {
			outputFile_ << results_[j].costFunctionValues[i] << ",";
			average += results_[j].costFunctionValues[i];
		}
		outputFile_ << average/numberOfRuns_;
		outputFile_ << "\n";
	}
}

void Simulator::writeHeader() {
	for(int i = 1; i <= numberOfRuns_; i++) {
		outputFile_ << "run" << std::to_string(i) << ",";
	}
	outputFile_ << "average";
	outputFile_ << std::endl;
}


}