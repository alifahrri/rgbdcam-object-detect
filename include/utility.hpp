#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <chrono>
#include <thread>

namespace utility {

using namespace std;
typedef chrono::duration<double, std::milli> milliseconds;

struct timer
{
    double elapsed_time = 0.0;
    double diff_time = 0.0;
    chrono::high_resolution_clock::time_point init_time = chrono::high_resolution_clock::now();

    inline double elapsed()
    {
        auto now = chrono::high_resolution_clock::now();
        auto last = elapsed_time;
        milliseconds dt;
        dt = now - init_time;
        elapsed_time = dt.count();
        diff_time = elapsed_time - last;
        return elapsed_time;
    }

    inline void sleep(double ms)
    {
        auto sleep_time = ms - elapsed();
        if(sleep_time>0)
            std::this_thread::sleep_for(std::chrono::milliseconds((int)sleep_time));
    }
};

}

#endif // UTILITY_HPP
