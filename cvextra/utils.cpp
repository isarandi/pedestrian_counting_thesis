#include "utils.hpp"
#include <stddef.h>
#include <ctime>
#include <string>

using namespace std;

string cvx::timestamp()
{
    time_t rawtime;
    std::time(&rawtime);

    size_t const bufSize = 80;
    char buffer[bufSize];

    std::strftime(buffer, bufSize, "%Y_%m_%dT%H_%M_%S", std::localtime(&rawtime));
    return string(buffer);
}

//void cvx::parallel_for_(Range range, std::function<void(Range)> function)
//{
//    cv::parallel_for_(range, cvx::details::ParallelBodyWithLambda(function));
//}
