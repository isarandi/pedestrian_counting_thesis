#include "stdx.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace stdx;

string stdx::to_string(double number, int nDecimalsAfterPoint)
{
    stringstream ss;
    ss << std::fixed
       << std::setprecision(nDecimalsAfterPoint)
       << std::showpoint
       << number;

    return ss.str();
}

string stdx::to_string(bool b)
{
    return b ? "true" : "false";
}

