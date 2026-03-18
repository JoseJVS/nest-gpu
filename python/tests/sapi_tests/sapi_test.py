import logging
import sys
import typing
from argparse import ArgumentParser
from functools import reduce
from traceback import format_exc

LOG = logging.getLogger(__name__)

import numpy as np
from mpi4py import MPI

import nestgpu

parser = ArgumentParser()
parser.add_argument("--rng_seed", type=int, default=12345)
parser.add_argument("--verbosity", type=int, default=20)
args = parser.parse_args()


def update_verbosity():
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s:\n%(message)s"))
    logging.basicConfig(level=args.verbosity, handlers=(stdout,))


def middle_factors(n: int) -> tuple:
    step = 2 if n % 2 else 1
    upper_bound = int(np.floor(np.sqrt(n))) + 1
    lower_factor = 1
    higher_factor = n
    for i in range(1, upper_bound, step):
        if n % i == 0:
            lower_factor, higher_factor = i, n // i
    return lower_factor, higher_factor


def main() -> None:
    nestgpu.set_rng_seed(args.rng_seed)
    nestgpu.SetBoolParam("check_node_maps", True)

    local_rank = nestgpu.HostId()
    num_processes = nestgpu.HostNum()
    width, length = middle_factors(num_processes)
    total_nodes = 3 * num_processes

    LOG.info("RANK %i: generating %ix%i tile grid", local_rank, width, length)
    nestgpu.generate_tile_grid(
        (0, 0),
        (1, 1),
        "Rectangle",
        (0.1,),
        tuple(),
        [{0} for r in range(num_processes)],
        0,
    )

    LOG.info("RANK %i: generating %i nodes in grid", local_rank, total_nodes)
    sp_ns = nestgpu.generate_nodes_in_grid("iaf_psc_alpha", total_nodes)

    LOG.info("RANK %i: computing spatial connections", local_rank)
    conn_index = nestgpu.compute_spatial_connections(
        sp_ns,
        sp_ns,
        {
            "mask_blueprint_name": "Circular",
            "mask_blueprint_params": (1,),
        },
        {
            "edge_wrap": True,
            "only_neighborhood": False,
            "allow_self_connections": False,
            "allow_multiplicity": False,
            "partition_connections_by_source": True,
            "conn_gen_name": "PairWiseBernoulli",
            "weight_df_name": "Distance",
            "weight_ufs_names": ["LowerBound", "Inverse"],
            "weight_ufs_params": [[0.0001, 0.0001], []],
            "delay_df_name": "Distance",
            "delay_ufs_names": ["Offset"],
            "delay_ufs_params": [[0.1]],
            "prob_df_name": "Constant",
            "prob_df_params": [1.0],
            "prob_ufs_names": [],
            "prob_ufs_params": [],
        },
    )

    LOG.info("RANK %i: calibrating network", local_rank)
    nestgpu.Calibrate()

    LOG.info("RANK %i: checking local connections", local_rank)
    if sp_ns.local_index is not None and sp_ns.local_length is not None:
        ns = nestgpu.NodeSeq(sp_ns.local_index, sp_ns.local_length)
        conn_list = nestgpu.GetConnections(ns, ns)
        conn_status = nestgpu.GetConnectionStatus(conn_list)

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

        spatial_conns = nestgpu.get_spatial_connections(conn_index)

        spatial_conn_map = {}
        for conn_part in spatial_conns[1][local_rank]:
            for conn_t in conn_part:
                source, target, weight, delay = conn_t
                if source in spatial_conn_map:
                    source_map = spatial_conn_map[source]
                    assert target not in source_map
                    source_map[target] = (weight, delay)
                else:
                    source_map = spatial_conn_map[source] = {}
                    source_map[target] = (weight, delay)

        assert len(gpu_conn_map) == len(
            spatial_conn_map
        ), f"{len(gpu_conn_map)} vs {len(spatial_conn_map)}"

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
        main()
        nestgpu.MpiFinalize()
    except Exception:
        LOG.critical(format_exc())
        MPI.COMM_WORLD.Abort(1)
