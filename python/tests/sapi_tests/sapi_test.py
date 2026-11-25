import json
import logging
import sys
import typing
from argparse import ArgumentParser
from functools import reduce
from traceback import format_exc

LOG = logging.getLogger(__name__)

# Nodes per process for a single process run, scaled by the number of processes
NODES_PER_PROCESS = 40

import numpy as np
from mpi4py import MPI

import nestgpu

parser = ArgumentParser()
parser.add_argument("--no_space", action="store_true")
parser.add_argument("--rng_seed", type=int, default=12345)
parser.add_argument("--verbosity", type=int, default=20)
args = parser.parse_args()


def update_verbosity():
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s:\n%(message)s"))
    logging.basicConfig(level=args.verbosity, handlers=(stdout,))


def no_space_main() -> None:
    nestgpu.SetKernelStatus({"rnd_seed": args.rng_seed, "check_node_maps": True})

    local_rank = nestgpu.HostId()
    num_processes = nestgpu.HostNum()
    local_ns = None
    nodes = []

    if 1 < num_processes:
        for i in range(num_processes):
            nodes.append(nestgpu.RemoteCreate(i, "iaf_psc_exp", 100).node_seq)
        local_ns = nodes[local_rank]
    else:
        local_ns = nestgpu.Create("iaf_psc_exp", 100)

    if 1 < num_processes:
        for i in range(num_processes):
            for j in range(num_processes):
                nestgpu.RemoteConnect(
                    i,
                    nodes[i],
                    j,
                    nodes[j],
                    {"rule": "fixed_indegree", "indegree": 10},
                    {"weight": 1, "delay": 1},
                )
    else:
        nestgpu.Connect(
            local_ns,
            local_ns,
            {"rule": "fixed_indegree", "indegree": 10},
            {"weight": 1, "delay": 1},
        )

    nestgpu.Calibrate()

    conn_list = nestgpu.GetConnections(local_ns, local_ns)
    conn_status = nestgpu.GetConnectionStatus(conn_list)
    print(json.dumps(conn_status, indent=4))
    print(len(conn_status))


def main() -> None:
    nestgpu.set_rng_seed(args.rng_seed)
    nestgpu.SetBoolParam("check_node_maps", True)

    local_rank = nestgpu.HostId()
    num_processes = nestgpu.HostNum()
    # Node density grows with the number of processes, so that each rank keeps
    # a high chance of having local connections
    total_nodes = NODES_PER_PROCESS * num_processes * num_processes

    LOG.info("RANK %i: tile grid", local_rank)
    nestgpu.generate_tile_grid(
        {
            "grid_origin": (0, 0),
            "grid_dimensions": (1, 1),
            "tile_type": "rectangle",
            "tile_side_lengths": (1,),
        },
        rank_tile_ownership="shared",
    )

    LOG.info("RANK %i: generating %i nodes in grid", local_rank, total_nodes)
    sp_ns = nestgpu.generate_nodes_in_grid(
        "iaf_psc_exp", total_nodes, grid_distribution_mode="balanced"
    )

    LOG.info("RANK %i: computing spatial connections", local_rank)
    conn_index = nestgpu.compute_spatial_connections(
        sp_ns,
        sp_ns,
        {
            "mask_blueprint_name": "circular",
            "mask_blueprint_params": (0.1,),
        },
        {
            "edge_wrap": True,
            "only_neighborhood": False,
            "allow_self_connections": False,
            "allow_multiplicity": False,
            "partition_connections": True,
            "rule": "pairwise_bernoulli",
            "weight_df_name": "constant",
            "weight_df_params": [42],
            "delay_df_name": "distance",
            "delay_ufs_names": ["offset"],
            "delay_ufs_params": [[42]],
            "prob_df_name": "constant",
            "prob_df_params": [0.2],
        },
    )

    # Annular mask: closed ring r_in <= distance <= r_out, disjoint from the
    # circular mask above (radius 0.1) so that the GPU check below can tell
    # the two sets apart by their weight. The delay is the distance (no offset).
    assert "annular" in nestgpu.get_parameter_names()["mask_shapes"]
    r_in, r_out = 0.12, 0.2
    LOG.info("RANK %i: computing annular spatial connections", local_rank)
    ring_index = nestgpu.compute_spatial_connections(
        sp_ns,
        sp_ns,
        {
            "mask_blueprint_name": "annular",
            "mask_blueprint_params": (r_in, r_out),
        },
        {
            "edge_wrap": True,
            "only_neighborhood": False,
            "allow_self_connections": False,
            "allow_multiplicity": False,
            "partition_connections": True,
            "rule": "pairwise_bernoulli",
            "weight_df_name": "constant",
            "weight_df_params": [43],
            "delay_df_name": "distance",
            "prob_df_name": "constant",
            "prob_df_params": [0.2],
        },
    )

    LOG.info("RANK %i: calibrating network", local_rank)
    nestgpu.Calibrate()

    LOG.info("RANK %i: checking local connections", local_rank)
    if sp_ns.local_index is not None and sp_ns.local_length is not None:
        local_conns = []
        for index in (conn_index, ring_index):
            incoming_conns, _ = nestgpu.get_spatial_connections(index)
            # Ranks without local connections have no entry
            local_conns.extend(incoming_conns.get(local_rank, []))
        # Every annular connection (weight 43) lies within the closed ring; the
        # delay is the (wrapped) distance
        eps = 1e-5
        ring_conns = 0
        for _, _, weights, delays in local_conns:
            for weight, delay in zip(weights, delays):
                if weight == 43:
                    ring_conns += 1
                    assert r_in - eps <= delay <= r_out + eps, delay
        # The ring check must not pass without checking anything
        assert not local_conns or 0 < ring_conns, "no annular connections"
        print(json.dumps(local_conns, indent=4))
        total_cons = 0
        spatial_conn_map = {}
        for sources, targets, weights, delays in local_conns:
            total_cons += len(sources)
            for i, source in enumerate(sources):
                target = targets[i]
                if source in spatial_conn_map:
                    assert target not in spatial_conn_map[source]
                    spatial_conn_map[source][target] = (weights[i], delays[i])
                else:
                    spatial_conn_map[source] = {target: (weights[i], delays[i])}

        if total_cons == 0:
            # Nothing to compare, and the GPU may hold no connections to retrieve
            LOG.warning("RANK %i: no local connections, skipping check", local_rank)
            return

        ns = nestgpu.NodeSeq(sp_ns.local_index, sp_ns.local_length)
        conn_list = nestgpu.GetConnections(ns, ns)
        conn_status = nestgpu.GetConnectionStatus(conn_list)
        print(json.dumps(conn_status, indent=4))

        gpu_conn_map = {}
        for conn_d in conn_status:
            source = conn_d["source"]
            target = conn_d["target"]
            weight = conn_d["weight"]
            delay = conn_d["delay"]
            if source in gpu_conn_map:
                source_map = gpu_conn_map[source]
                assert target not in source_map
                source_map[target] = (weight, delay)
            else:
                source_map = gpu_conn_map[source] = {}
                source_map[target] = (weight, delay)

        assert len(spatial_conn_map) == len(gpu_conn_map), (
            f"sapi: {len(spatial_conn_map)}, gpu: {len(gpu_conn_map)}"
        )

        for source, target_map in gpu_conn_map.items():
            assert source in spatial_conn_map
            sp_target_map = spatial_conn_map[source]
            assert len(target_map) == len(sp_target_map)
            for target, (weight, delay) in target_map.items():
                assert target in sp_target_map
                sp_weight, sp_delay = sp_target_map[target]
                computed_sp_delay = float(
                    int(np.round(np.float32(sp_delay) / np.float32(0.1)))
                    * np.float32(0.1)
                )  # Conversion to mimic delay discretization in kernel
                assert weight == sp_weight and np.isclose(delay, computed_sp_delay)


if __name__ == "__main__":
    update_verbosity()
    try:
        nestgpu.ConnectMpiInit()
        if args.no_space:
            no_space_main()
        else:
            main()
        nestgpu.MpiFinalize()
    except Exception:
        LOG.critical(format_exc())
        MPI.COMM_WORLD.Abort(1)
