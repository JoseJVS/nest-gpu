#ifndef TIMER_REGISTER_H
#define TIMER_REGISTER_H

#include <chrono>
#include <string>
#include <utility>
#include <unordered_map>


namespace sapi
{
typedef std::unordered_map< std::string, double > TimerData;


struct StopWatch
{
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::nanoseconds duration_;

    void start()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        duration_ += std::chrono::high_resolution_clock::now() - start_;
    }

    double time() const
    {
        return duration_.count() / 1e9;
    }
};


struct TimerRegister
{
    std::unordered_map< std::string, StopWatch > registered_timers_;

    StopWatch* get_register_timer( std::string&& name )
    {
        auto reg = registered_timers_.find( name );
        if ( reg == registered_timers_.end() )
        {
            reg = registered_timers_.emplace(
                std::make_pair(
                    std::move( name ),
                    StopWatch()
                )
            ).first;
        }
        return &reg->second;
    }

    std::string to_string( uint8_t tabs = 0 ) const
    {
        std::string res = "";
        std::string t_str = "";
        for ( ; tabs > 0; --tabs )
            t_str += "\t";
        for ( const auto& [name, timer] : registered_timers_ )
            res += t_str + name + ": " + std::to_string( timer.time() ) + "\n";
        return res;
    }

    TimerData to_map() const
    {
        TimerData map;
        for ( const auto& [name, timer] : registered_timers_ )
            map.emplace(
                std::make_pair(
                    std::string( name ),
                    timer.time()
                )
            );
        return map;
    }
};
}


#endif
