#ifndef CONNECTION_GENERATORS_H
#define CONNECTION_GENERATORS_H

#include <functional>

#include "mask_containers.h"
#include "random_generators.h"
#include "connection_functor.h"
#include "connection_containers.h"


namespace sapi
{
template < typename CoordT >
class ConnectionGenerator : public Clonable< ConnectionGenerator< CoordT > >
{
public:
    ConnectionGenerator() = default;
    ConnectionGenerator( const ConnectionGenerator& ) = delete;
    ConnectionGenerator( ConnectionGenerator&& ) = default;

    void set_total_num_connections( const mult_t& );
    void set_cf_collection( CFCollection< CoordT >&& );

    virtual bool is_initialized() const = 0;

    virtual void compute_connections(
        LeafConnectionInfo& lci,
        ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
        nest::RngPtr const& rng,
        const bool& allow_multiplicity,
        const bool& allow_self_connections
    ) const = 0;

protected:
    mult_t tnc_ = 0;
    CFCollection< CoordT > cfc_;
};


template < typename CoordT >
inline void ConnectionGenerator< CoordT >::set_total_num_connections(
    const mult_t& tnc
)
{
    if ( tnc < 0 )
        throw std::invalid_argument( "Cannot set a negative total connection count for connection generators" );
    tnc_ = tnc;
}


template < typename CoordT >
inline void ConnectionGenerator< CoordT >::set_cf_collection(
    CFCollection< CoordT >&& cfc
)
{
    cfc_ = std::move( cfc );
}


template < typename CoordT >
class ProbabilisticCG : public ConnectionGenerator< CoordT >
{
public:
    ProbabilisticCG() = default;
    ProbabilisticCG( const ProbabilisticCG& ) = delete;
    ProbabilisticCG( ProbabilisticCG&& ) = default;

    bool is_initialized() const override;

    virtual mult_t draw_connection_multiplicity(
        const space_t& probability_value,
        nest::RngPtr const& rng,
        const bool& allow_multiplicity
    ) const = 0;

    void compute_connections(
        LeafConnectionInfo& lci,
        ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
        nest::RngPtr const& rng,
        const bool& allow_multiplicity,
        const bool& allow_self_connections
    ) const override;
};


template < typename CoordT >
inline bool ProbabilisticCG< CoordT >::is_initialized() const
{
    return this->cfc_.weight_functor_.is_initialized() &&
        this->cfc_.delay_functor_.is_initialized() &&
        this->cfc_.probability_functor_.is_initialized();
}


template < typename CoordT >
inline bool _remove_self_target(
    const nodeidx_t& driver_index,
    std::map< nodeidx_t, Displacement< CoordT > >& pool_map,
    const bool& allow_self_connections
)
{
    if ( allow_self_connections )
        return false;

    if ( const auto self_search =
        pool_map.find( driver_index );
        self_search != pool_map.end() )
    {
        pool_map.erase( self_search );
        if ( pool_map.empty() )
            return true;
    }

    return false;
}


template < typename CoordT >
void ProbabilisticCG< CoordT >::compute_connections(
    LeafConnectionInfo& lci,
    ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
    nest::RngPtr const& rng,
    const bool& allow_multiplicity,
    const bool& allow_self_connections
) const
{
    assert(
        is_initialized() &&
        rng != nullptr &&
        lci.connection_map_.empty() &&
        !displacement_map.empty()
    );

    const bool monitor_connection_count = 0 < this->tnc_;

    const std::function< bool( const mult_t& ) > keep_connecting =
        monitor_connection_count
        ? []( const mult_t& count )
        { return 0 < count; }
        : []( const mult_t& )
        { return true; };

    const std::function< mult_t( mult_t&, mult_t&& ) > adjust_multiplicity =
        monitor_connection_count
        ? []( mult_t& count, mult_t&& multiplicity )
        {
            const bool adjust = count < multiplicity;
            multiplicity = adjust
            ? count
            : multiplicity;
            count -= adjust
            ? count
            : multiplicity;
            return multiplicity;
        }
        : []( mult_t&, mult_t&& multiplicity )
        { return multiplicity; };

    for ( auto& [driver_index, pool_map] : displacement_map )
    {
        assert( !pool_map.empty() );

        if ( _remove_self_target(
            driver_index, pool_map, allow_self_connections
        ) ) continue;

        auto max_num_connections = this->tnc_;
        std::unordered_map< nodeidx_t, ConnectionInfo > temp_target_map;
        for ( auto& [pool_index, displacement] : pool_map )
        {
            if ( !keep_connecting( max_num_connections ) )
                break;

            const auto multiplicity = adjust_multiplicity(
                max_num_connections,
                draw_connection_multiplicity(
                    this->cfc_.probability_functor_( displacement ),
                    rng,
                    allow_multiplicity
                )
            );
            if ( multiplicity < 1 ) continue;

            lci.total_generated_connections_ += multiplicity;

            temp_target_map.emplace(
                std::make_pair(
                    nodeidx_t( pool_index ),
                    std::make_tuple(
                        this->cfc_.weight_functor_( displacement ),
                        this->cfc_.delay_functor_( displacement ),
                        mult_t( multiplicity )
                    )
                )
            );
        }

        if ( !temp_target_map.empty() )
            lci.connection_map_.emplace(
                std::make_pair(
                    nodeidx_t( driver_index ),
                    std::move( temp_target_map )
                )
            );

        pool_map.clear();
    }

    // Check overflow
    assert( 0 <= lci.total_generated_connections_ );

    displacement_map.clear();
}


template < typename CoordT >
class PairWiseBernoulliPCG : public ProbabilisticCG< CoordT >
{
public:
    PairWiseBernoulliPCG() = default;
    PairWiseBernoulliPCG( const PairWiseBernoulliPCG& ) = delete;
    PairWiseBernoulliPCG( PairWiseBernoulliPCG&& ) = default;

    std::unique_ptr< ConnectionGenerator< CoordT > >
        clone() const override;

    mult_t draw_connection_multiplicity(
        const space_t& probability_value,
        nest::RngPtr const& rng,
        const bool& allow_multiplicity
    ) const override;
};


template < typename CoordT >
inline std::unique_ptr< ConnectionGenerator< CoordT > >
PairWiseBernoulliPCG< CoordT >::clone() const
{
    auto pbcg = std::make_unique< PairWiseBernoulliPCG< CoordT > >();
    pbcg->tnc_ = this->tnc_;
    pbcg->cfc_ = this->cfc_;
    return pbcg;
}


template < typename CoordT >
inline mult_t
PairWiseBernoulliPCG< CoordT >::draw_connection_multiplicity(
    const space_t& probability_value,
    nest::RngPtr const& rng,
    const bool&
) const
{
    return rng->drand() < static_cast< nest::uniform_real_distribution::result_type >( probability_value );
}


template < typename CoordT >
class PairWisePoissonPCG : public ProbabilisticCG< CoordT >
{
public:
    PairWisePoissonPCG() = default;
    PairWisePoissonPCG( const PairWisePoissonPCG& ) = delete;
    PairWisePoissonPCG( PairWisePoissonPCG&& ) = default;

    std::unique_ptr< ConnectionGenerator< CoordT > >
        clone() const override;

    mult_t draw_connection_multiplicity(
        const space_t& probability_value,
        nest::RngPtr const& rng,
        const bool& allow_multiplicity
    ) const override;

protected:
    mutable nest::poisson_distribution poisson_distribution_;
};


template < typename CoordT >
inline std::unique_ptr< ConnectionGenerator< CoordT > >
PairWisePoissonPCG< CoordT >::clone() const
{
    auto pbcg = std::make_unique< PairWisePoissonPCG< CoordT > >();
    pbcg->tnc_ = this->tnc_;
    pbcg->cfc_ = this->cfc_;
    return pbcg;
}


template < typename CoordT >
inline mult_t
PairWisePoissonPCG< CoordT >::draw_connection_multiplicity(
    const space_t& probability_value,
    nest::RngPtr const& rng,
    const bool& allow_multiplicity
) const
{
    poisson_distribution_.param(
        static_cast< nest::poisson_distribution::param_type >( probability_value )
    );
    return allow_multiplicity
        ? static_cast< mult_t >( poisson_distribution_( rng ) )
        : static_cast< mult_t >( 0 < poisson_distribution_( rng ) );
}


template < typename CoordT >
class FixedNumberCG : public ConnectionGenerator< CoordT >
{
public:
    FixedNumberCG() = default;
    FixedNumberCG( const FixedNumberCG& ) = delete;
    FixedNumberCG( FixedNumberCG&& ) = default;

    bool is_initialized() const override;

    std::unique_ptr< ConnectionGenerator< CoordT > > clone() const override;

    void compute_connections(
        LeafConnectionInfo& lci,
        ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
        nest::RngPtr const& rng,
        const bool& allow_multiplicity,
        const bool& allow_self_connections
    ) const;

    std::vector< mult_t > generate_connection_counts(
        const std::size_t& available_targets,
        nest::RngPtr const& rng,
        const bool& allow_multiplicity
    ) const;
};


template < typename CoordT >
inline bool FixedNumberCG< CoordT >::is_initialized() const
{
    return this->cfc_.weight_functor_.is_initialized()
        && this->cfc_.delay_functor_.is_initialized()
        && !this->cfc_.probability_functor_.is_initialized()
        && 0 <= this->tnc_;
}


template < typename CoordT >
inline std::unique_ptr< ConnectionGenerator< CoordT > > FixedNumberCG< CoordT >::clone() const
{
    auto fncg = std::make_unique< FixedNumberCG< CoordT > >();
    fncg->tnc_ = this->tnc_;
    fncg->cfc_ = this->cfc_;
    return fncg;
}


template < typename CoordT >
void FixedNumberCG< CoordT >::compute_connections(
    LeafConnectionInfo& lci,
    ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
    nest::RngPtr const& rng,
    const bool& allow_multiplicity,
    const bool& allow_self_connections
) const
{
    assert(
        is_initialized() &&
        rng != nullptr &&
        lci.connection_map_.empty() &&
        !displacement_map.empty()
    );

    if ( this->tnc_ == 0 )
    {
        displacement_map.clear();
        return;
    }

    for ( auto& [driver_index, pool_map] : displacement_map )
    {
        assert( !pool_map.empty() );

        if ( _remove_self_target(
            driver_index, pool_map, allow_self_connections
        ) ) continue;

        const auto driver_emplace_it = lci.connection_map_.emplace(
            std::make_pair(
                nodeidx_t( driver_index ),
                std::unordered_map< nodeidx_t, ConnectionInfo >()
            )
        ).first;

        const auto connection_counts = generate_connection_counts(
            pool_map.size(),
            rng,
            allow_multiplicity
        );

        auto connection_count_it = connection_counts.cbegin();
        for ( auto& [pool_index, displacement] : pool_map )
        {
            const auto multiplicity = *connection_count_it++;
            if ( multiplicity < 1 ) continue;

            lci.total_generated_connections_ += multiplicity;

            driver_emplace_it->second.emplace(
                std::make_pair(
                    nodeidx_t( pool_index ),
                    std::make_tuple(
                        this->cfc_.weight_functor_( displacement ),
                        this->cfc_.delay_functor_( displacement ),
                        mult_t( multiplicity )
                    )
                )
            );
        }

        pool_map.clear();
    }

    // Check overflow
    assert( 0 <= lci.total_generated_connections_ );

    displacement_map.clear();
}


template < typename CoordT >
std::vector< mult_t >
FixedNumberCG< CoordT >::generate_connection_counts(
    const std::size_t& available_targets,
    nest::RngPtr const& rng,
    const bool& allow_multiplicity
) const
{
    assert( 0 < available_targets );

    std::vector< mult_t > connection_counts;
    auto connections_todo = this->tnc_;

    if ( static_cast< std::size_t >( this->tnc_ ) < available_targets )
        // More available targets than fixed number
        connection_counts.resize( available_targets, 0 );
    else
    {
        if ( available_targets == 1 )
        {
            connection_counts.resize( 1, allow_multiplicity ? this->tnc_ : 1 );
            connections_todo = 0;
        }
        else
        {
            connection_counts.resize( available_targets, !allow_multiplicity );
            connections_todo = allow_multiplicity * this->tnc_;
        }
    }

    if ( 0 < connections_todo && allow_multiplicity )
    {
        nest::uniform_int_distribution t_idx_dist;
        t_idx_dist.param(
            nest::uniform_int_distribution::param_type(
                0, static_cast< nest::uniform_int_distribution::result_type >(
                    available_targets - 1
                    )
            )
        );
        while ( 0 < connections_todo )
        {
            ++connection_counts[ t_idx_dist( rng ) ];
            --connections_todo;
        }
    }
    else if ( 0 < connections_todo )
    {
        std::vector< nodeidx_t > indexes( available_targets );
        std::iota( indexes.begin(), indexes.end(), 0 );
        rng->shuffle( indexes );

        // Due to previous tests here available targets is guaranteed > connections_todo
        while ( 0 < connections_todo )
        {
            ++connection_counts[ indexes[ connections_todo ] ];
            --connections_todo;
        }
    }

    // Here we test that at least one connection count is assigned
    assert( connections_todo < this->tnc_ );

    return connection_counts;
}
}


#endif
