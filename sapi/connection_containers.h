#ifndef CONNECTION_CONTAINERS_H
#define CONNECTION_CONTAINERS_H

#include <tuple>
#include <unordered_map>

#include "sapi_config.h"


namespace sapi
{
// Connection weight, connection delay, connection multiplicity
typedef std::tuple< conn_t, conn_t, mult_t > ConnectionInfo;
typedef std::unordered_map< nodeidx_t,
    std::unordered_map< nodeidx_t, ConnectionInfo > > NodeConnectionMap;


struct LeafConnectionInfo
{
    count_t total_generated_connections_ = 0;

    NodeConnectionMap connection_map_;

    LeafConnectionInfo() = default;
    LeafConnectionInfo( const LeafConnectionInfo& ) = delete;
    LeafConnectionInfo( LeafConnectionInfo&& ) = default;
    ~LeafConnectionInfo() = default;

    bool operator==( const LeafConnectionInfo& lci ) const
    {
        return connection_map_ == lci.connection_map_;
    }
};


// Maps are aggregated by driver tile/node
struct TileConnectionInfo
{
    count_t total_generated_connections_ = 0;

    std::unordered_map < tileidx_t,
        std::unordered_map < tileidx_t, LeafConnectionInfo > > aggregated_connection_map_;

    NodeConnectionMap consolidated_connection_map_;

    TileConnectionInfo() = default;
    TileConnectionInfo( const TileConnectionInfo& ) = delete;
    TileConnectionInfo( TileConnectionInfo&& ) = default;
    ~TileConnectionInfo() = default;

    bool operator==( const TileConnectionInfo& tci ) const
    {
        return aggregated_connection_map_ == tci.aggregated_connection_map_ &&
            consolidated_connection_map_ == tci.consolidated_connection_map_;
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
