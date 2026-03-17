/*
 *  timer_manager.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TIMER_MANAGER_H
#define TIMER_MANAGER_H

#include <chrono>
#include <string>
#include <unordered_map>

#include "thread_aligned_array.h"


namespace sapi
{
struct StopWatch
{
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::nanoseconds duration_;

    void start();

    void stop();

    double time() const;
};


inline void StopWatch::start()
{
    start_ = std::chrono::high_resolution_clock::now();
}


inline void StopWatch::stop()
{
    duration_ += std::chrono::high_resolution_clock::now() - start_;
}


inline double StopWatch::time() const
{
    return duration_.count() / 1e9;
}


struct TimerRegister final : public Cloneable< TimerRegister >
{
    std::unordered_map< std::string, StopWatch > registered_timers_;

    StopWatch* get_register_timer( std::string&& name );

    std::unordered_map< std::string, double > to_map() const;

    std::unique_ptr< TimerRegister > clone() const override;
};


inline StopWatch* TimerRegister::get_register_timer( std::string&& name )
{
    auto search = registered_timers_.find( name );
    if ( search == registered_timers_.end() )
        search = registered_timers_.emplace(
            std::make_pair(
                std::move( name ),
                StopWatch()
            )
        ).first;
    return &search->second;
}


inline std::unique_ptr< TimerRegister > TimerRegister::clone() const
{
    return std::make_unique< TimerRegister >();
}


struct RecordedTimes
{
    std::unordered_map< std::string, double > rank_times_;
    std::unordered_map< std::string, std::vector< double > > thread_times_;

    std::string to_string( uint8_t tabs = 0 ) const;
};


class TimerManager
{
public:
    TimerManager();
    TimerManager( const TimerManager& ) = delete;
    TimerManager( TimerManager&& ) = default;
    ~TimerManager() = default;

    void initialize();

    bool is_initialized() const;

    TimerRegister* get_rank_registry() const;

    TimerRegister* get_thread_registry() const;

    RecordedTimes get_times() const;

protected:
    bool initialized_ = false;
    std::unique_ptr< TimerRegister > rank_registry_;
    TAArray< TimerRegister > thread_registers_;
};


inline void TimerManager::initialize()
{
    rank_registry_ = std::make_unique< TimerRegister >();
    thread_registers_.clone( TimerRegister() );
    initialized_ = true;
}


inline bool TimerManager::is_initialized() const
{
    return initialized_;
}


inline TimerRegister* TimerManager::get_rank_registry() const
{
    assert( initialized_ );
    return rank_registry_.get();
}


inline TimerRegister* TimerManager::get_thread_registry() const
{
    assert( initialized_ );
    return thread_registers_.get_thread_item( get_thread_num() );
}
}


#endif
