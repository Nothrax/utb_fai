#include <simulator/function/FunctionFactory.hpp>
#include <simulator/function/Schweffel.hpp>
#include <simulator/function/DeJong1St.hpp>
#include <simulator/function/DeJong2Nd.hpp>
#include <simulator/function/Rastrigin.hpp>

namespace simulator::function{


std::shared_ptr<IFunction> FunctionFactory::makeFunction(FunctionEnum functionEnum) {
	switch(functionEnum) {

		case FunctionEnum::SCHWEFFEL:
			return std::make_shared<Schweffel>();
		case FunctionEnum::DEJONG_1:
			return std::make_shared<DeJong1st>();
		case FunctionEnum::DEJONG_2:
			return std::make_shared<DeJong2Nd>();
		case FunctionEnum::RASTRIGIN:
			return std::make_shared<Rastrigin>();
	}
	return nullptr;
}
}