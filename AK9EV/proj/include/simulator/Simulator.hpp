#pragma once

#include <simulator/function/IFunction.hpp>

#include <filesystem>
#include <fstream>

namespace simulator{

enum class AlgorithmType{
	RANDOM_SEARCH = 0,
	HILL_CLIMBING,
	PSU
};

class Simulator {
public:

	void simulate(const std::string& path, int numberOfRuns, int dimensionCount, int fes, function::FunctionEnum functionType, AlgorithmType algoType);
	~Simulator();
private:
	struct Run{
		int runNumber;
		std::vector<double> costFunctionValues;
	};

	std::ofstream outputFile_;
	int dimensionCount_;
	int fes_;
	int numberOfSteps_;
	int numberOfRuns_;
	int actualRun_;
	std::shared_ptr<function::IFunction> function_;

	std::vector<Run> results_;

	void randomSearch();
	void hillClimbing();
	void psu();

	void writeHeader();
	void writeRuns();
};
}