#pragma once

#include <simulator/function/IFunction.hpp>

namespace simulator::function{

class FunctionFactory {
public:
	FunctionFactory() = delete;

	/**
	 * Get function name and return function instance
	 * @param functionName
	 * @return
	 */
	static std::shared_ptr<IFunction> makeFunction(FunctionEnum functionEnum);
};
}
