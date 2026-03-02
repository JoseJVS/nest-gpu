/*
 *  grid_neighborhood.h
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

#ifndef GRID_NEIGHBORHOOD_H
#define GRID_NEIGHBORHOOD_H

#include "tile_grid.h"


namespace sapi
{
// Forward definition to link with vp_interface
vp_t get_mpi_rank();
vp_t get_num_mpi_processes();


struct GridNeighborhood
{
    using TileSet = std::vector< std::set< tileidx_t > >::const_iterator;
    using RankSet = std::vector< std::set< vp_t > >::const_iterator;

    const vp_t local_rank_;
    const vp_t num_processes_;
    bool has_owners_ = false;

    // Special condition where
    // one rank only owns one tile
    // each tile is only owned by one rank
    bool rank_tile_bijection_ = false;

    // Iterator to set of locally owned tiles
    TileSet local_owned_tiles_;
    // Iterator to local rank MPI neighborhood
    RankSet local_rank_neighbors_;
    // Tile to owning ranks ( one-to-one mapping on number of tiles in grid )
    std::vector< std::set< vp_t > > tile_ranks_ownership_map_;
    // Rank to owned tiles ( one-to-one mapping on number of ranks in world )
    std::vector< std::set< tileidx_t > > rank_tiles_ownership_map_;
    // Rank to rank neighborhood ( one-to-one mapping on number of ranks in world )
    // ( based on owned tile neighborhood )
    std::vector< std::set< vp_t > > rank_neighbors_map_;

    GridNeighborhood()
        : local_rank_( get_mpi_rank() )
        , num_processes_( get_num_mpi_processes() )
    {
    }

    GridNeighborhood( const GridNeighborhood& ) = delete;
    GridNeighborhood( GridNeighborhood&& ) = default;

    GridNeighborhood(
        const vp_t& local_rank,
        const vp_t& num_processes
    )
        : local_rank_( local_rank )
        , num_processes_( num_processes )
    {
    }

    template < typename CoordT >
    void set_tile_ownership(
        std::vector< std::set< tileidx_t > >&&
        rank_tiles_ownership_map,
        const TileGrid< CoordT >& tile_grid
    );

    std::string to_string() const
    {
        assert( has_owners_ );
        std::string res = "Tile ranks ownership:\n";
        for ( std::size_t tix = 0; tix < tile_ranks_ownership_map_.size(); ++tix )
        {
            res += "{ " + std::to_string( tix ) + " : (";
            for ( const auto& rix : tile_ranks_ownership_map_[ tix ] )
                res += " " + std::to_string( rix ) + ",";
            res += ") }, ";
        }

        res += "\nRank tiles ownership:\n";
        for ( vp_t rix = 0; rix < num_processes_; ++rix )
        {
            res += "{ " + std::to_string( rix ) + " : (";
            for ( const auto& tix : rank_tiles_ownership_map_[ rix ] )
                res += " " + std::to_string( tix ) + ",";
            res += ") }, ";
        }

        res += "\nRank neighborhood:\n";
        for ( vp_t rix = 0; rix < num_processes_; ++rix )
        {
            res += "{ " + std::to_string( rix ) + " : (";
            for ( const auto& nix : rank_neighbors_map_[ rix ] )
                res += " " + std::to_string( nix ) + ",";
            res += ") }, ";
        }

        return res;
    }
};


template < typename CoordT >
void GridNeighborhood::set_tile_ownership(
    std::vector< std::set< tileidx_t > >&&
    rank_tiles_ownership_map,
    const TileGrid< CoordT >& tile_grid
)
{
    if ( has_owners_ )
        throw std::runtime_error( "Tile ownership already defined" );
    if ( tile_grid.positions_.empty() )
        throw std::invalid_argument( "Tile ownership can only be defined with an initialized grid" );
    if ( rank_tiles_ownership_map.size() != static_cast< std::size_t >( num_processes_ ) )
        throw std::invalid_argument( "Ownership has to be defined for all ranks" );

    vp_t curr_rank = 0;
    vp_t injective_ranks = 0;
    std::vector< std::set< vp_t > > tile_ranks_ownership_map( tile_grid.num_tiles_ );
    for ( const auto& tile_set : rank_tiles_ownership_map )
    {
        injective_ranks += tile_set.size() == 1;
        for ( const auto& tile_idx : tile_set )
            tile_ranks_ownership_map.at( tile_idx ).insert( curr_rank );
        ++curr_rank;
    }

    tileidx_t injective_tiles = 0;
    auto tile_pos_it = tile_grid.positions_.cbegin();
    std::vector< std::set< vp_t > > rank_neighbors_map( num_processes_ );
    for ( const auto& source_rank_set : tile_ranks_ownership_map )
    {
        injective_tiles += source_rank_set.size() == 1;
        for ( const auto& source_rank : source_rank_set )
        {
            const auto neighborhood_it = rank_neighbors_map.begin() + source_rank;
            for ( const auto& neighbor_index : tile_pos_it->tile_neighborhood_ )
            {
                for ( const auto& target_rank : tile_ranks_ownership_map[ neighbor_index ] )
                {
                    if ( source_rank == target_rank ) continue;
                    neighborhood_it->insert( target_rank );
                }
            }
        }
        ++tile_pos_it;
    }

    rank_tile_bijection_ = injective_ranks == num_processes_ && injective_tiles == tile_grid.num_tiles_;
    rank_tiles_ownership_map_ = std::move( rank_tiles_ownership_map );
    tile_ranks_ownership_map_ = std::move( tile_ranks_ownership_map );
    rank_neighbors_map_ = std::move( rank_neighbors_map );
    local_owned_tiles_ = rank_tiles_ownership_map_.cbegin() + local_rank_;
    local_rank_neighbors_ = rank_neighbors_map_.cbegin() + local_rank_;
    has_owners_ = true;
}
}


#endif
