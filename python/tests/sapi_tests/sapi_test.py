import logging
import sys
from argparse import ArgumentParser
from functools import reduce
from hashlib import sha256
from pathlib import Path
from traceback import format_exc

LOG = logging.getLogger(__name__)

import numpy as np
from mpi4py import MPI

import nestgpu

parser = ArgumentParser()
parser.add_argument("--tile_splits", type=int, default=-1)
parser.add_argument("--edge_wrap", action="store_true")
parser.add_argument("--total_nodes", type=int, default=1000)
parser.add_argument("--rand_pos", action="store_true")
parser.add_argument("--fixed_conn", action="store_true")
parser.add_argument("--conn_chance", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--cutoff", type=float, default=5)
parser.add_argument("--rng_seed", type=int, default=12345)
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--output_prefix", type=str, default="nest_comparison")
parser.add_argument("--output_format", type=str, default=".pdf")
parser.add_argument("--verbosity", type=int, default=20)
args = parser.parse_args()

local_rank = MPI.COMM_WORLD.Get_rank()
num_processes = MPI.COMM_WORLD.Get_size()

sqrt2 = np.sqrt(2)
sqrt_procs = int(np.floor(np.sqrt(num_processes)))
sq_grid = sqrt_procs**2 == num_processes
log4 = np.log(4)
log2 = np.log(2)


def compute_num_splits(tile_type: str, num_nodes: int, num_tiles: int) -> int:
    if 0 <= args.tile_splits:
        return args.tile_splits
    match tile_type:
        case "Square":
            return int(np.max(np.floor(np.log(num_nodes / (10 * num_tiles)) / log4), 0))
        case "Triangle":
            return int(np.max(np.floor(np.log(num_nodes / (10 * num_tiles)) / log2), 0))
        case "Hexagon":
            return int(
                np.max(np.floor(np.log(num_nodes / (60 * num_tiles)) / log2 + 1), 0)
            )
        case _:
            raise ValueError("Incorrect tile type")


def main() -> None:
    nestgpu.set_rng_seed(args.rng_seed)

    if sq_grid and 1 < num_processes:
        LOG.info("RANK %i SAPI: generating %ix%i tile grid", local_rank, sqrt_procs)
        coord = -1.0 + 1.0 / sqrt_procs
        radius = sqrt2 / sqrt_procs
        nestgpu.generate_tile_grid(
            (coord, coord),
            (sqrt_procs, sqrt_procs),
            "Square",
            (radius, 0),
            [{r} for r in range(num_processes)],
            compute_num_splits("Square", args.total_nodes, num_processes),
            args.edge_wrap,
        )
    else:
        LOG.info(
            "RANK %i SAPI: generating single tile grid",
            local_rank,
        )
        nestgpu.generate_tile_grid(
            (0, 0),
            (1, 1),
            "Square",
            (np.sqrt(2), 0),
            [{0} for _ in range(num_processes)],
            compute_num_splits("Square", args.total_nodes, 1),
            args.edge_wrap,
        )

    sp_ns = nestgpu.generate_nodes_in_grid("iaf_psc_alpha", args.total_nodes)
    conn_index = nestgpu.compute_spatial_connections(
        sp_ns,
        sp_ns,
        {
            "mask_blueprint_name": "Circular",
            "mask_blueprint_params": (sqrt2,),
        },
        {
            "edge_wrap": args.edge_wrap,
            "only_neighborhood": False,
            "allow_self_connections": False,
            "allow_multiplicity": False,
            "conn_gen_name": "PairWiseBernoulli",
            "weight_df_name": "Distance",
            "delay_df_name": "Constant",
            "prob_df_name": "Constant" if args.fixed_conn else "Distance",
            "prob_df_params": [args.conn_chance] if args.fixed_conn else [],
            "prob_ufs_names": [] if args.fixed_conn else ["Exponential"],
            "prob_ufs_params": [] if args.fixed_conn else [[args.beta]],
        },
    )

    nestgpu.Calibrate()


if __name__ == "__main__":
    try:
        nestgpu.ConnectMpiInit()
        main()
        nestgpu.MpiFinalize()
    except Exception:
        LOG.critical(format_exc())
        MPI.COMM_WORLD.Abort(1)
