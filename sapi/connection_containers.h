#ifndef CONNECTION_CONTAINERS_H
#define CONNECTION_CONTAINERS_H

#include <map>
#include <tuple>
#include <vector>
#include <forward_list>
#include <unordered_map>
#include <cassert>

#include "sapi_config.h"


namespace sapi
{
// Connection source, target, weight, connection delay, connection multiplicity
typedef std::tuple< conn_index_t, conn_index_t, conn_param_t, conn_param_t, mult_t > ConnectionInfo;


struct ConnectionVectors
{
    count_t sizes_ = 0;

    std::vector< conn_index_t > connection_sources_;
    std::vector< conn_index_t > connection_targets_;
    std::vector< conn_param_t > connection_weights_;
    std::vector< conn_param_t > connection_delays_;

    ConnectionVectors() = default;
    ConnectionVectors( const ConnectionVectors& ) = delete;
    ConnectionVectors( ConnectionVectors&& ) = default;

    void prepare_vectors( const count_t& size )
    {
        assert(
            0 <= size &&
            connection_sources_.empty() &&
            connection_targets_.empty() &&
            connection_weights_.empty() &&
            connection_delays_.empty()
        );

        if ( size == 0 )
            return;

        connection_sources_.reserve( size );
        connection_targets_.reserve( size );
        connection_weights_.reserve( size );
        connection_delays_.reserve( size );

        sizes_ = size;
    }
};


// Maps are aggregated by driver tile/node
struct TileConnectionInfo
{
    count_t total_generated_connections_ = 0;

    std::forward_list< std::forward_list< ConnectionInfo > > procedural_connection_list_;

    std::vector< ConnectionVectors > source_unique_connection_vectors_;

    TileConnectionInfo() = default;
    TileConnectionInfo( const TileConnectionInfo& ) = delete;
    TileConnectionInfo( TileConnectionInfo&& ) = default;
    ~TileConnectionInfo() = default;

    bool operator==( const TileConnectionInfo& tci ) const
    {
        return build_connection_map() == tci.build_connection_map();
    }

    std::map< conn_index_t,
        std::map< conn_index_t,
        std::tuple< conn_param_t, conn_param_t, mult_t > > >
        build_connection_map() const
    {
        std::map< conn_index_t,
            std::map< conn_index_t,
            std::tuple< conn_param_t, conn_param_t, mult_t > > > conn_map;

        for ( const auto& conn_vec : source_unique_connection_vectors_ )
        {
            for ( count_t conn_idx = 0; conn_idx < conn_vec.sizes_; ++conn_idx )
            {
                const auto source = conn_vec.connection_sources_[ conn_idx ];
                const auto target = conn_vec.connection_targets_[ conn_idx ];
                const auto weight = conn_vec.connection_weights_[ conn_idx ];
                const auto delay = conn_vec.connection_delays_[ conn_idx ];

                auto source_search = conn_map.find( source );
                if ( source_search == conn_map.end() )
                    source_search = conn_map.emplace(
                        std::make_pair(
                            source,
                            std::map< conn_index_t,
                            std::tuple< conn_param_t, conn_param_t, mult_t > >()
                        )
                    ).first;

                auto target_search = source_search->second.find( target );
                if ( target_search == source_search->second.end() )
                    target_search = source_search->second.emplace(
                        std::make_pair(
                            target,
                            std::make_tuple(
                                weight,
                                delay,
                                1
                            )
                        )
                    ).first;
                else
                    std::get< 2 >( target_search->second ) += 1;
            }
        }

        return conn_map;
    }
};


struct RankConnectionInfo
{
    std::unordered_map< vp_t, TileConnectionInfo > outgoing_connections_;
    std::unordered_map< vp_t, TileConnectionInfo > incoming_connections_;

    RankConnectionInfo() = default;
    RankConnectionInfo( const RankConnectionInfo& ) = delete;
    RankConnectionInfo( RankConnectionInfo&& rci ) = default;
    ~RankConnectionInfo() = default;

    bool operator==( const RankConnectionInfo& rci ) const
    {
        return outgoing_connections_ == rci.outgoing_connections_
            && incoming_connections_ == rci.incoming_connections_;
    }
};
}


#endif
