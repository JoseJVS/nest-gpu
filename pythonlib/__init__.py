#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Spatial API for generating spatially defined node graphs.
Nodes can be queried and interconnected using spatial connection procedures.
Parallelized using MPI and OpenMP

Authors: JoseJVS

"""

__all__ = [
    "compute_spatial_connections",
    "generate_nodes_in_grid",
    "generate_tile_grid",
    "get_distributed_node_sequences",
    "get_grid_vertices",
    "get_nodes",
    "get_num_processes",
    "get_num_threads",
    "get_rank",
    "get_rng_seed",
    "get_rng_type",
    "get_spatial_connections",
    "get_timer_data",
    "insert_positions_in_grid",
    "reset_api",
    "set_num_threads",
    "set_rng_seed",
    "set_rng_type",
]


from . import ll_sapi
from .ll_sapi import (
    compute_spatial_connections,
    generate_nodes_in_grid,
    generate_tile_grid,
    get_distributed_node_sequences,
    get_grid_vertices,
    get_nodes,
    get_num_processes,
    get_num_threads,
    get_rank,
    get_rng_seed,
    get_rng_type,
    get_spatial_connections,
    get_timer_data,
    insert_positions_in_grid,
    reset_api,
    set_num_threads,
    set_rng_seed,
    set_rng_type,
)

ll_sapi.init()
