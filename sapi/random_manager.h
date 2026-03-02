/*
 *  random_manager.h
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

/*
 *  Adapted from NEST::random_manager.h
 *
 *  Under copyright:
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef RANDOM_MANAGER_H
#define RANDOM_MANAGER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <cassert>

#include "sapi_config.h"

// Third party
#include "random_generators.h"
#include "pcg_random.hpp"


namespace sapi
{
constexpr static const char* const DEFAULT_RNG_TYPE_ = "pcg64";
constexpr static const uint32_t DEFAULT_BASE_SEED_ = 143202461;
constexpr static const uint32_t RANK_SEEDER_ = 0xc229212d;
constexpr static const uint32_t THREAD_SEEDER_ = 0x37722d5e;
constexpr static const uint32_t PARITY_SEEDER_ = 0xb84c9bae;


// Forward definition to link with vp_interface
vp_t get_mpi_rank();
vp_t get_num_mpi_processes();
vp_t get_thread_num();
vp_t get_max_omp_threads();


class RandomManager
{
public:
    const uint32_t local_rank_;
    const vp_t num_processes_;

    RandomManager( const RandomManager& ) = delete;
    RandomManager( RandomManager&& ) = delete;

    RandomManager()
        : local_rank_( static_cast< uint32_t >( get_mpi_rank() ) )
        , num_processes_( get_num_mpi_processes() )
    {
        _register_known_rng_types();
        initialize();
    }

    RandomManager( const vp_t& local_rank, const vp_t& num_processes )
        : local_rank_( static_cast< uint32_t >( local_rank ) )
        , num_processes_( num_processes )
    {
        _register_known_rng_types();
        initialize();
    }

    ~RandomManager()
    {
        finalize();
        for ( const auto& rng_pair : registered_rng_types_ )
            delete rng_pair.second;
    }

    void initialize();
    void finalize();

    bool is_initialized() const;

    nest::RngPtr get_rank_synced_rng() const;
    nest::RngPtr get_rank_specific_rng() const;
    nest::RngPtr get_tid_synced_rng( const vp_t& tid ) const;
    nest::RngPtr get_tid_specific_rng( const vp_t& tid ) const;

    nest::RngPtr reseed_rank_paired_rng(
        const vp_t& tid,
        const vp_t& target_rank,
        const bool& inverted_source_target,
        const tileidx_t& tile_idx,
        const tileidx_t& leaf_idx
    ) const;

    void update_rank_paired_seed( const vp_t& target_rank );

    template < typename RNG >
    void register_rng_type( std::string&& name );

    void set_rng_seed( const uint32_t& seed );

    void set_rng_type( const std::string& name );

protected:
    void _register_known_rng_types();

    bool initialized_ = false;
    vp_t num_threads_;

    std::string current_rng_type_ = DEFAULT_RNG_TYPE_;
    nest::BaseRandomGeneratorFactory* current_rng_factory_ = nullptr;
    std::unordered_map< std::string, nest::BaseRandomGeneratorFactory* >
        registered_rng_types_;

    nest::RngPtr rank_synced_rng_ = nullptr;
    nest::RngPtr rank_specific_rng_ = nullptr;
    std::vector< nest::RngPtr > tid_synced_rng_vec_;
    std::vector< nest::RngPtr > tid_specific_rng_vec_;
    std::vector< nest::RngPtr > tid_rank_paired_rng_vec_;

    uint32_t base_seed_ = DEFAULT_BASE_SEED_;
    std::vector< uint32_t > rank_paired_seeds_;
};


inline void RandomManager::_register_known_rng_types()
{
    register_rng_type< std::mt19937 >(
        "mt19937"
    );
    register_rng_type< std::mt19937_64 >(
        "mt19937_64"
    );
    register_rng_type< pcg32_k1024_fast >(
        "pcg32"
    );
    register_rng_type< pcg64_k1024_fast >(
        "pcg64"
    );

    current_rng_factory_ = registered_rng_types_.at( std::string( DEFAULT_RNG_TYPE_ ) );
}


inline void RandomManager::initialize()
{
    if (
        !rank_paired_seeds_.empty() ||
        !tid_synced_rng_vec_.empty() ||
        !tid_specific_rng_vec_.empty() ||
        !tid_rank_paired_rng_vec_.empty()
        )
        finalize();

    rank_paired_seeds_.resize( num_processes_, 0 );

    rank_synced_rng_ = current_rng_factory_->create(
        { base_seed_, RANK_SEEDER_ }
    );
    rank_specific_rng_ = current_rng_factory_->create(
        { base_seed_, RANK_SEEDER_, PARITY_SEEDER_, local_rank_ }
    );

    num_threads_ = get_max_omp_threads();
    tid_synced_rng_vec_.resize( num_threads_, nullptr );
    tid_specific_rng_vec_.resize( num_threads_, nullptr );
    tid_rank_paired_rng_vec_.resize( num_threads_, nullptr );

#pragma omp parallel default( none )\
    shared( tid_synced_rng_vec_, tid_specific_rng_vec_, tid_rank_paired_rng_vec_ )\
    firstprivate( base_seed_, local_rank_, current_rng_factory_ )
    {
        const auto tid = static_cast< uint32_t >( get_thread_num() );
        tid_synced_rng_vec_[ tid ] = current_rng_factory_->create(
            { base_seed_, THREAD_SEEDER_, tid }
        );
        tid_specific_rng_vec_[ tid ] = current_rng_factory_->create(
            { base_seed_, THREAD_SEEDER_, PARITY_SEEDER_, local_rank_, tid }
        );
        tid_rank_paired_rng_vec_[ tid ] = current_rng_factory_->create(
            { base_seed_ } // First seed does not really matter for these rngs
        );
    }

    initialized_ = true;
}


inline void RandomManager::finalize()
{
    delete rank_synced_rng_;
    delete rank_specific_rng_;
    for ( const auto& vec : {
        &tid_synced_rng_vec_,
        &tid_specific_rng_vec_,
        &tid_rank_paired_rng_vec_
        } )
    {
        for ( const auto& rng : *vec )
            delete rng;
        vec->clear();
    }
    rank_paired_seeds_.clear();
    initialized_ = false;
}


inline bool RandomManager::is_initialized() const
{
    return initialized_;
}


inline nest::RngPtr RandomManager::get_rank_synced_rng() const
{
    assert( initialized_ );
    return rank_synced_rng_;
}


inline nest::RngPtr RandomManager::get_rank_specific_rng() const
{
    assert( initialized_ );
    return rank_specific_rng_;
}


inline nest::RngPtr RandomManager::get_tid_synced_rng( const vp_t& tid ) const
{
    assert( initialized_ && 0 <= tid && tid < num_threads_ );
    return tid_synced_rng_vec_[ tid ];
}


inline nest::RngPtr RandomManager::get_tid_specific_rng( const vp_t& tid ) const
{
    assert( initialized_ && 0 <= tid && tid < num_threads_ );
    return tid_specific_rng_vec_[ tid ];
}


inline nest::RngPtr RandomManager::reseed_rank_paired_rng(
    const vp_t& tid,
    const vp_t& target_rank,
    const bool& inverted_source_target,
    const tileidx_t& tile_idx,
    const tileidx_t& leaf_idx
) const
{
    assert(
        initialized_ && 0 <= tid && tid < num_threads_ &&
        0 <= target_rank && target_rank < num_processes_
    );
    const auto rng = tid_rank_paired_rng_vec_[ tid ];
    rng->reseed(
        {
            base_seed_,
            THREAD_SEEDER_,
            PARITY_SEEDER_,
            inverted_source_target ? static_cast< uint32_t >( target_rank ) : local_rank_,
            inverted_source_target ? local_rank_ : static_cast< uint32_t >( target_rank ),
            rank_paired_seeds_[ target_rank ],
            static_cast< uint32_t >( tile_idx ),
            static_cast< uint32_t >( leaf_idx )
        }
    );
    return rng;
}


inline void RandomManager::update_rank_paired_seed(
    const vp_t& target_rank
)
{
    assert( 0 <= target_rank && target_rank < num_processes_ );
#pragma omp atomic
    ++rank_paired_seeds_[ target_rank ];
}


inline void RandomManager::set_rng_seed( const uint32_t& seed )
{
    if ( seed < 0 || 1u << 31 < seed )
        throw std::invalid_argument( "Invalid RNG seed" );

    base_seed_ = seed;
    if ( initialized_ )
        finalize();
    initialize();
}


template < typename RNG >
inline void RandomManager::register_rng_type( std::string&& name )
{
    if ( registered_rng_types_.find( name ) != registered_rng_types_.end() )
        throw std::invalid_argument( name + " is already registered" );
    else
        registered_rng_types_.emplace(
            std::make_pair(
                std::move( name ),
                new nest::RandomGeneratorFactory< RNG >()
            )
        );
}


inline void RandomManager::set_rng_type( const std::string& name )
{
    if ( name == current_rng_type_ )
    {
        if ( initialized_ )
            return;
        initialize();
    }

    const auto search = registered_rng_types_.find( name );
    if ( search == registered_rng_types_.end() )
        throw std::invalid_argument( "RNG type not known by random manager" );

    current_rng_factory_ = search->second;
    current_rng_type_ = name;

    if ( initialized_ )
        finalize();

    initialize();
}
}


#endif
