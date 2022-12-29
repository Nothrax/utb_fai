#include <iostream>

#include <simulator/Simulator.hpp>

int main() {
	using namespace simulator;


	try{
		Simulator sim;
		sim.simulate("../output/psu_dejong1_10dim",30, 10,5000,function::FunctionEnum::DEJONG_1, AlgorithmType::PSU);

		sim.simulate("../output/psu_dejong1_30dim",30, 30,5000,function::FunctionEnum::DEJONG_1, AlgorithmType::PSU);

        sim.simulate("../output/psu_dejong2_10dim",30, 10,5000,function::FunctionEnum::DEJONG_2, AlgorithmType::PSU);
        sim.simulate("../output/psu_dejong2_30dim",30, 30,5000,function::FunctionEnum::DEJONG_2, AlgorithmType::PSU);

        sim.simulate("../output/psu_schweffel_10dim",30, 10,5000,function::FunctionEnum::SCHWEFFEL, AlgorithmType::PSU);
        sim.simulate("../output/psu_schweffel_30dim",30, 30,5000,function::FunctionEnum::SCHWEFFEL, AlgorithmType::PSU);

        sim.simulate("../output/psu_rastrigin_10dim",30, 10,5000,function::FunctionEnum::RASTRIGIN, AlgorithmType::PSU);
        sim.simulate("../output/psu_rastrigin_30dim",30, 30,5000,function::FunctionEnum::RASTRIGIN, AlgorithmType::PSU);


	}catch(std::exception &e){
		std::cerr << e.what() << std::endl;
	}catch(...){
		std::cerr << "Unknown exception" << std::endl;
	}
	return 0;
}
