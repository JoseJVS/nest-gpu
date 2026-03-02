/*
 *  spatial_containers.h
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

#ifndef SPATIAL_CONTAINERS_H
#define SPATIAL_CONTAINERS_H

#include <string>
#include <vector>
#include <unordered_map>

#include "sapi_config.h"


namespace sapi
{
// This map is used to export tile vertices
typedef std::unordered_map< tileidx_t,
    std::pair< std::vector< std::vector< space_t > >,
    std::unordered_map< tileidx_t,
    std::vector< std::vector< space_t > > > >
> GridTileVertexMap;

typedef std::unordered_map< nodeidx_t,
    std::vector< space_t > > NodeIdxAnyCoordMap;

// This map is used to export tiled node coordinates
// regardless of coordinate structure type used
typedef std::unordered_map< tileidx_t,
    NodeIdxAnyCoordMap > TileIdxNodeIdxACM;


// This map is used to export tiled node coordinates
// using tile and leaf sub tile index
typedef std::unordered_map< tileidx_t,
    TileIdxNodeIdxACM > NestedTileIdxNodeIdxACM;


// ----- Input parameters -----
struct MaskParameters
{
    // At least one required
    std::string mask_blueprint_name_;
    std::vector< space_t > mask_blueprint_params_;
    std::vector< space_t > mask_blueprint_offset_;
    std::string source_mask_name_;
    std::vector< space_t > source_mask_params_;
    std::vector< space_t > source_mask_offset_;
    std::string target_mask_name_;
    std::vector< space_t > target_mask_params_;
    std::vector< space_t > target_mask_offset_;
};


struct ConnectionParameters
{
    // Control parameters
    bool edge_wrap_ = false;
    bool only_neighborhood_ = false;
    bool inverted_conn_rule_ = false;
    bool allow_multiplicity_ = false;
    bool allow_self_connections_ = false;
    mult_t total_number_connections_ = 0;

    // Connection generation
    std::string conn_gen_name_;

    // Weight computation
    std::string weight_df_name_;
    std::vector< space_t > weight_df_params_;
    std::vector< std::string > weight_ufs_names_;
    std::vector< std::vector< space_t > > weight_ufs_params_;

    // Delay computation
    std::string delay_df_name_;
    std::vector< space_t > delay_df_params_;
    std::vector< std::string > delay_ufs_names_;
    std::vector< std::vector< space_t > > delay_ufs_params_;

    // Probability drawing
    std::string prob_df_name_;
    std::vector< space_t > prob_df_params_;
    std::vector< std::string > prob_ufs_names_;
    std::vector< std::vector< space_t > > prob_ufs_params_;
};
}


#endif
