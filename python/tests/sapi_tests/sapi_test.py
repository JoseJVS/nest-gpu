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
parser.add_argument("--tile_splits", type=int, default=-1)
parser.add_argument("--edge_wrap", action="store_true")
parser.add_argument("--total_nodes", type=int, default=1000)
parser.add_argument("--fixed_conn", action="store_true")
parser.add_argument("--conn_chance", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--rng_seed", type=int, default=12345)
parser.add_argument("--verbosity", type=int, default=20)
args = parser.parse_args()


def middle_factors(n: int) -> typing.Tuple[int, int]:
    step = 2 if n % 2 else 1
    factors = reduce(
        list.__add__,
        ([i, n // i] for i in range(1, int(np.sqrt(n)) + 1, step) if n % i == 0),
    )
    # Should be at least 4 and a multiple of 2 as duplicates are not removed and pairs of factors are inserted
    if (
        len(factors) < 4
    ):  # Under above premise only 1 and N are in list if this condition is true
        return 1, n
    return factors[-2], factors[-1]


def update_verbosity():
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s:\n%(message)s"))
    logging.basicConfig(level=args.verbosity, handlers=(stdout,))


def compute_num_splits(tile_type: str, num_nodes: int, num_tiles: int) -> int:
    if 0 <= args.tile_splits:
        return args.tile_splits
    match tile_type:
        case "Square":
            return int(
                np.max(np.floor(np.log(num_nodes / (10 * num_tiles)) / np.log(4)), 0)
            )
        case "Triangle":
            return int(
                np.max(np.floor(np.log(num_nodes / (10 * num_tiles)) / np.log(2)), 0)
            )
        case "Hexagon":
            return int(
                np.max(
                    np.floor(np.log(num_nodes / (60 * num_tiles)) / np.log(2) + 1), 0
                )
            )
        case _:
            raise ValueError("Incorrect tile type")


def main() -> None:
    nestgpu.set_rng_seed(args.rng_seed)
    nestgpu.SetBoolParam("check_node_maps", False)

    local_rank = nestgpu.HostId()
    num_processes = nestgpu.HostNum()
    width, length = middle_factors(num_processes)

    LOG.info("RANK %i: generating %ix%i tile grid", local_rank, width, length)
    nestgpu.generate_tile_grid(
        (0, 0),
        (width, length),
        "Square",
        (0.5, 0),
        [{r} for r in range(num_processes)],
        compute_num_splits("Square", args.total_nodes, 1),
        args.edge_wrap,
    )

    LOG.info("RANK %i: generating %i nodes in grid", local_rank, args.total_nodes)
    sp_ns = nestgpu.generate_nodes_in_grid("iaf_psc_alpha", args.total_nodes)

    LOG.info("RANK %i: computing spatial connections", local_rank)
    conn_index = nestgpu.compute_spatial_connections(
        sp_ns,
        sp_ns,
        {
            "mask_blueprint_name": "Circular",
            "mask_blueprint_params": (1,),
        },
        {
            "edge_wrap": args.edge_wrap,
            "only_neighborhood": False,
            "allow_self_connections": False,
            "allow_multiplicity": False,
            "conn_gen_name": "PairWiseBernoulli",
            "weight_df_name": "Distance",
            "weight_ufs_names": ["LowerBound", "Inverse"],
            "weight_ufs_params": [[0.0001, 0.0001], []],
            "delay_df_name": "Distance",
            "delay_ufs_names": ["Offset"],
            "delay_ufs_params": [[0.1]],
            "prob_df_name": "Constant" if args.fixed_conn else "Distance",
            "prob_df_params": [args.conn_chance] if args.fixed_conn else [],
            "prob_ufs_names": [] if args.fixed_conn else ["Exponential"],
            "prob_ufs_params": [] if args.fixed_conn else [[args.beta]],
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
        for conn_t in spatial_conns[1][local_rank]:
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
