#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CTypes bindings for low level Spatial API.

Authors: JoseJVS.

"""

import ctypes
import pathlib
import sys
import typing

vp_t: typing.TypeAlias = ctypes.c_int32
tix_t: typing.TypeAlias = ctypes.c_int32
nix_t: typing.TypeAlias = ctypes.c_int32
lnix_t: typing.TypeAlias = ctypes.c_int64
conn_t: typing.TypeAlias = ctypes.c_float
mult_t: typing.TypeAlias = ctypes.c_uint16
split_t: typing.TypeAlias = ctypes.c_uint8
space_t: typing.TypeAlias = ctypes.c_double
CData: typing.TypeAlias = ctypes._SimpleCData | ctypes.Structure | ctypes._Pointer


def pair_template(t0: type[CData], t1: type[CData]) -> type[ctypes.Structure]:
    class PairT(ctypes.Structure):
        _fields_ = [("first_", t0), ("second_", t1)]

    return PairT


def triplet_template(
    t0: type[CData],
    t1: type[CData],
    t2: type[CData],
) -> type[ctypes.Structure]:
    class TripletT(ctypes.Structure):
        _fields_ = [("first_", t0), ("second_", t1), ("third_", t2)]

    return TripletT


def array_template(t: type[CData]) -> type[ctypes.Structure]:
    class ArrayT(ctypes.Structure):
        _fields_ = [("size_", ctypes.c_size_t), ("array_", ctypes.POINTER(t))]
        array_type: type[CData] = t

        def resize(self, size: int) -> None:
            if size < 0:
                raise ValueError("Incorrect resize size")
            self.size_ = ctypes.c_size_t(size)
            if size > 0:
                self.array_ = (self.array_type * size)()

    return ArrayT


def pair_array_template(k: type[CData], v: type[CData]) -> type[ctypes.Structure]:
    return array_template(pair_template(k, v))


OptionalIndex = pair_template(ctypes.c_bool, ctypes.c_size_t)
CharArray = array_template(ctypes.c_char)
SpaceTArray = array_template(space_t)
TileIdxArray = array_template(tix_t)

NestedCharArray = array_template(CharArray)
NestedSpaceTArray = array_template(SpaceTArray)
NestedTileIdxArray = array_template(TileIdxArray)

TiledNodeSequencePairArray = pair_array_template(
    vp_t, pair_array_template(tix_t, pair_template(nix_t, nix_t))
)
ConnectionPairArray = pair_array_template(
    vp_t,
    pair_array_template(
        nix_t,
        pair_array_template(
            nix_t,
            triplet_template(conn_t, conn_t, mult_t),
        ),
    ),
)
NodeCoordPairArray = pair_array_template(
    tix_t, pair_array_template(nix_t, array_template(space_t))
)
NestedNodeCoordPairArray = pair_array_template(tix_t, NodeCoordPairArray)
GridTileVerticesPairArray = pair_array_template(
    tix_t,
    pair_template(NestedSpaceTArray, pair_array_template(tix_t, NestedSpaceTArray)),
)
TimerDataPairArray = pair_array_template(CharArray, ctypes.c_double)


def check_bool(b_val: ctypes.c_bool) -> None:
    if not b_val:
        raise ValueError("is false")


def check_ptr(ptr: ctypes._Pointer) -> None:
    if not bool(ptr):
        raise ValueError("is nullptr")


def safe_convert_to_c(
    c_type: type[ctypes._SimpleCData], p_val: int | float | bool
) -> ctypes._SimpleCData:
    c_val = c_type(p_val)
    if c_val.value != p_val:
        raise ValueError("Py to C conversion error")
    return c_val


def safe_convert_to_py(
    p_type: type[int | float | bool], c_val: ctypes._SimpleCData
) -> int | float | bool:
    p_val = p_type(c_val)
    if p_val != c_val:
        raise ValueError("C to Py conversion error")
    return p_val


c_data_to_py_int = lambda val: safe_convert_to_py(p_type=int, c_val=val)
c_data_to_py_float = lambda val: safe_convert_to_py(p_type=float, c_val=val)
c_data_to_py_bool = lambda val: safe_convert_to_py(p_type=bool, c_val=val)


def str_to_carr(string: str) -> ctypes.Structure:
    ca = CharArray()
    ca.resize(len(string) + 1)
    if ca.size_ > 1:
        ca.array_ = ctypes.create_string_buffer(
            string.encode("utf-8", "strict") + b"\0"
        )
    else:
        ca.array_[0] = b"\0"
    return ca


def carr_to_str(carr: ctypes.Structure) -> str:
    if carr.size_ < 2:
        return str()
    else:
        check_ptr(carr.array_)
        return (
            bytes()
            .join(carr.array_[i] for i in range(carr.size_))
            .decode("utf-8", "strict")
            .rstrip("\0")
        )


def str_col_to_nested_carr(str_col: typing.Collection[str]) -> ctypes.Structure:
    nca = NestedCharArray()
    nca.resize(len(str_col))
    for i, string in enumerate(str_col):
        nca.array_[i] = str_to_carr(string)
    return nca


def nested_carr_to_str_col(nested_carr: ctypes.Structure) -> typing.List[str]:
    if nested_carr.size_ < 1:
        return list()
    else:
        check_ptr(nested_carr.array_)
        return [carr_to_str(nested_carr.array_[i]) for i in range(nested_carr.size_)]


def num_col_to_num_arr(
    num_arr_type: type[ctypes.Structure], num_col: typing.Collection[int | float]
) -> ctypes.Structure:
    arr = num_arr_type()
    arr.resize(len(num_col))
    for i, v in enumerate(num_col):
        arr.array_[i] = safe_convert_to_c(arr.array_type, v)
    return arr


def num_arr_to_num_col(
    num_type: type[int | float], num_arr: ctypes.Structure
) -> typing.List[int | float]:
    if num_arr.size_ < 1:
        return list()
    else:
        check_ptr(num_arr.array_)
        return [
            safe_convert_to_py(num_type, num_arr.array_[i])
            for i in range(num_arr.size_)
        ]


def nested_num_col_to_nested_arr(
    num_arr_type: type[ctypes.Structure],
    nested_num_arr_type: type[ctypes.Structure],
    nested_num_col: typing.Collection[typing.Collection[int | float]],
) -> ctypes.Structure:
    na = nested_num_arr_type()
    na.resize(len(nested_num_col))
    for i, nit in enumerate(nested_num_col):
        na.array_[i] = num_col_to_num_arr(num_arr_type, nit)
    return na


def nested_num_arr_to_nested_num_col(
    num_type: type[int | float],
    nested_num_arr: ctypes.Structure,
) -> typing.List[typing.List[int | float]]:
    if nested_num_arr.size_ < 1:
        return list()
    else:
        check_ptr(nested_num_arr.array_)
        return [
            num_arr_to_num_col(num_type, nested_num_arr.array_[i])
            for i in range(nested_num_arr.size_)
        ]


int_col_to_tia = lambda col: num_col_to_num_arr(num_arr_type=TileIdxArray, num_col=col)
tia_to_int_col = lambda arr: num_arr_to_num_col(num_type=int, num_arr=arr)
nested_int_col_to_nested_tia = lambda n_col: nested_num_col_to_nested_arr(
    num_arr_type=TileIdxArray,
    nested_num_arr_type=NestedTileIdxArray,
    nested_num_col=n_col,
)
nested_tia_to_nested_int_col = lambda n_arr: nested_num_arr_to_nested_num_col(
    num_type=int, nested_num_arr=n_arr
)
float_col_to_sta = lambda col: num_col_to_num_arr(num_arr_type=SpaceTArray, num_col=col)
sta_to_float_col = lambda arr: num_arr_to_num_col(num_type=float, num_arr=arr)
nested_float_col_to_nested_sta = lambda n_col: nested_num_col_to_nested_arr(
    num_arr_type=SpaceTArray,
    nested_num_arr_type=NestedSpaceTArray,
    nested_num_col=n_col,
)
nested_sta_to_nested_float_col = lambda n_arr: nested_num_arr_to_nested_num_col(
    num_type=float, nested_num_arr=n_arr
)


def dict_to_tiled_node_sequence_pair_array(
    d: typing.Dict[
        int,  # MPI rank
        typing.Dict[
            int,  # Tile index
            typing.Tuple[int, int],  # First node in sequence, length of sequence
        ],
    ],
) -> ctypes.Structure:  # TiledNodeSequencePairArray
    tnspa = TiledNodeSequencePairArray()
    tnspa.resize(len(d))
    for r, (rank, tix_ns_map) in enumerate(d.items()):
        rank_tns_parr = tnspa.array_[r]
        rank_tns_parr.first_ = safe_convert_to_c(vp_t, rank)
        rank_tns_parr.second_.resize(len(tix_ns_map))
        for t, (tix, ns) in enumerate(tix_ns_map.items()):
            tix_ns_pair = rank_tns_parr.second_.array_[t]
            tix_ns_pair.first_ = safe_convert_to_c(tix_t, tix)
            tix_ns_pair.second_.first_ = safe_convert_to_c(nix_t, ns[0])
            tix_ns_pair.second_.second_ = safe_convert_to_c(nix_t, ns[1])

    return tnspa


def tiled_node_sequence_pair_array_to_dict(
    tnspa: ctypes.Structure,  # TiledNodeSequencePairArray
) -> typing.Dict[
    int,  # MPI rank
    typing.Dict[
        int,  # Tile index
        typing.Tuple[int, int],  # First node in sequence, length of sequence
    ],
]:
    if tnspa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(tnspa.array_)
        for r in range(tnspa.size_):
            rank_tns_parr = tnspa.array_[r]
            rank = c_data_to_py_int(rank_tns_parr.first_)
            rank_dict = res[rank] = dict()
            if rank_tns_parr.second_.size_ > 0:
                check_ptr(rank_tns_parr.second_.array_)
                for t in range(rank_tns_parr.second_.size_):
                    tix_ns_pair = rank_tns_parr.second_.array_[t]
                    tix = c_data_to_py_int(tix_ns_pair.first_)
                    ns = (
                        c_data_to_py_int(tix_ns_pair.second_.first_),
                        c_data_to_py_int(tix_ns_pair.second_.second_),
                    )
                    rank_dict[tix] = ns

                if len(rank_dict) != rank_tns_parr.second_.size_:
                    raise ValueError("Corrupted tiled node sequence pair array")

            elif bool(rank_tns_parr.second_.array_):
                raise ValueError("Corrupted tiled node sequence pair array")

        if len(res) != tnspa.size_:
            raise ValueError("Corrupted tiled node sequence pair array")

        return res


def dict_to_connection_pair_array(
    d: typing.Dict[
        int,  # MPI rank
        typing.Dict[
            int,  # Source node index
            typing.Dict[
                int,  # Target node index
                typing.Tuple[
                    float, float, int
                ],  # Connection weight, delay, multiplicity
            ],
        ],
    ],
) -> ctypes.Structure:  # ConnectionPairArray
    cnnpa = ConnectionPairArray()
    cnnpa.resize(len(d))
    for r, (rank, source_conn_map) in enumerate(d.items()):
        rank_conn_parr = cnnpa.array_[r]
        rank_conn_parr.first_ = safe_convert_to_c(vp_t, rank)
        rank_conn_parr.second_.resize(len(source_conn_map))
        for s, (source_node, target_conn_map) in enumerate(source_conn_map.items()):
            source_conn_parr = rank_conn_parr.second_.array_[s]
            source_conn_parr.first_ = safe_convert_to_c(nix_t, source_node)
            source_conn_parr.second_.resize(len(target_conn_map))
            for t, (target_node, conn_info) in enumerate(target_conn_map.items()):
                target_conn_pair = source_conn_parr.second_.array_[t]
                target_conn_pair.first_ = safe_convert_to_c(nix_t, target_node)
                target_conn_pair.second_.first_ = safe_convert_to_c(
                    conn_t, conn_info[0]
                )
                target_conn_pair.second_.second_ = safe_convert_to_c(
                    conn_t, conn_info[1]
                )
                target_conn_pair.second_.third_ = safe_convert_to_c(
                    mult_t, conn_info[2]
                )

    return cnnpa


def connection_pair_array_to_dict(
    cnnpa: ctypes.Structure,  # ConnectionPairArray
) -> typing.Dict[
    int,  # MPI rank
    typing.Dict[
        int,  # Source node index
        typing.Dict[
            int,  # Target node index
            typing.Tuple[float, float, int],  # Connection weight, delay, multiplicity
        ],
    ],
]:
    if cnnpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(cnnpa.array_)
        for r in range(cnnpa.size_):
            rank_conn_parr = cnnpa.array_[r]
            rank = c_data_to_py_int(rank_conn_parr.first_)
            rank_dict = res[rank] = dict()
            if rank_conn_parr.second_.size_ > 0:
                check_ptr(rank_conn_parr.second_.array_)
                for s in range(rank_conn_parr.second_.size_):
                    source_conn_parr = rank_conn_parr.second_.array_[s]
                    source_node = c_data_to_py_int(source_conn_parr.first_)
                    target_dict = rank_dict[source_node] = dict()
                    if source_conn_parr.second_.size_ > 0:
                        check_ptr(source_conn_parr.second_.array_)
                        for t in range(source_conn_parr.second_.size_):
                            target_conn_pair = source_conn_parr.second_.array_[t]
                            target_node = c_data_to_py_int(target_conn_pair.first_)
                            conn_info = (
                                c_data_to_py_float(target_conn_pair.second_.first_),
                                c_data_to_py_float(target_conn_pair.second_.second_),
                                c_data_to_py_int(target_conn_pair.second_.third_),
                            )
                            target_dict[target_node] = conn_info

                        if len(target_dict) != source_conn_parr.second_.size_:
                            raise ValueError("Corrupted connection pair array")

                    elif bool(source_conn_parr.second_.array_):
                        raise ValueError("Corrupted connection pair array")

                if len(rank_dict) != rank_conn_parr.second_.size_:
                    raise ValueError("Corrupted connection pair array")

            elif bool(rank_conn_parr.second_.array_):
                raise ValueError("Corrupted connection pair array")

        if len(res) != cnnpa.size_:
            raise ValueError("Corrupted connection pair array")

        return res


def dict_to_node_coord_pair_array(
    d: typing.Dict[
        int, typing.Dict[int, typing.List[float]]
    ],  # Tile index : Node index : Coordinates
) -> ctypes.Structure:  # NodeCoordPairArray
    ncpa = NodeCoordPairArray()
    ncpa.resize(len(d))
    for t, (tix, node_coord_map) in enumerate(d.items()):
        tix_node_coord_parr = ncpa.array_[t]
        tix_node_coord_parr.first_ = safe_convert_to_c(tix_t, tix)
        tix_node_coord_parr.second_.resize(len(node_coord_map))
        for n, (node, coord_list) in enumerate(node_coord_map.items()):
            node_coord_pair = tix_node_coord_parr.second_.array_[n]
            node_coord_pair.first_ = safe_convert_to_c(nix_t, node)
            node_coord_pair.second_.resize(len(coord_list))
            for c, coord in enumerate(coord_list):
                node_coord_pair.second_.array_[c] = safe_convert_to_c(space_t, coord)

    return ncpa


def node_coord_pair_array_to_dict(
    ncpa: ctypes.Structure,  # NodeCoordPairArray
) -> typing.Dict[
    int, typing.Dict[int, typing.List[float]]  # Tile index : Node index : Coordinates
]:
    if ncpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(ncpa.array_)
        for t in range(ncpa.size_):
            tix_node_coord_parr = ncpa.array_[t]
            tix = c_data_to_py_int(tix_node_coord_parr.first_)
            tile_dict = res[tix] = dict()
            if tix_node_coord_parr.second_.size_ > 0:
                check_ptr(tix_node_coord_parr.second_.array_)
                for n in range(tix_node_coord_parr.second_.size_):
                    node_coord_pair = tix_node_coord_parr.second_.array_[n]
                    nix = c_data_to_py_int(node_coord_pair.first_)
                    if node_coord_pair.second_.size_ > 0:
                        check_ptr(node_coord_pair.second_.array_)
                        coord = [
                            c_data_to_py_float(node_coord_pair.second_.array_[c])
                            for c in range(node_coord_pair.second_.size_)
                        ]
                        tile_dict[nix] = coord
                    elif bool(node_coord_pair.second_.array_):
                        raise ValueError("Corrupted node coord pair array")

                if len(tile_dict) != tix_node_coord_parr.second_.size_:
                    raise ValueError("Corrupted node coord pair array")

            elif bool(tix_node_coord_parr.second_.array_):
                raise ValueError("Corrupted node coord pair array")

        if len(res) != ncpa.size_:
            raise ValueError("Corrupted node coord pair array")

        return res


def dict_to_nested_node_coord_pair_array(
    d: typing.Dict[
        int, typing.Dict[int, typing.Dict[int, typing.List[float]]]
    ],  # Tile index : Leaf index : Node index : Coordinates
) -> ctypes.Structure:  # NodeCoordPairArray
    nncpa = NestedNodeCoordPairArray()
    nncpa.resize(len(d))
    for t, (tix, leaf_node_coord_map) in enumerate(d.items()):
        tix_leaf_node_coord_parr = nncpa.array_[t]
        tix_leaf_node_coord_parr.first_ = safe_convert_to_c(tix_t, tix)
        tix_leaf_node_coord_parr.second_ = dict_to_node_coord_pair_array(
            leaf_node_coord_map
        )

    return nncpa


def nested_node_coord_pair_array_to_dict(
    nncpa: ctypes.Structure,  # NestedNodeCoordPairArray
) -> typing.Dict[
    int,  # Tile index
    typing.Dict[
        int,  # Leaf index
        typing.Dict[int, typing.List[float]],  # Node index : Coordinates
    ],
]:
    if nncpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(nncpa.array_)
        for t in range(nncpa.size_):
            tix_leaf_node_coord_parr = nncpa.array_[t]
            tix = c_data_to_py_int(tix_leaf_node_coord_parr.first_)
            res[tix] = node_coord_pair_array_to_dict(tix_leaf_node_coord_parr.second_)

        if len(res) != nncpa.size_:
            raise ValueError("Corrupted node coord pair array")

        return res


def dict_to_grid_tile_vertices_pair_array(
    d: typing.Dict[
        int,  # Tile index
        typing.Tuple[
            typing.List[typing.List[float]],  # Tile vertices
            typing.Dict[
                int,  # Sub tile index
                typing.List[typing.List[float]],  # Sub tile vertices
            ],
        ],
    ],
) -> ctypes.Structure:
    gtvpa = GridTileVerticesPairArray()
    gtvpa.resize(len(d))
    for i, (tix, vst_tuple) in enumerate(d.items()):
        tile_vstp_parr = gtvpa.array_[i]
        tile_vstp_parr.first_ = safe_convert_to_c(tix_t, tix)
        tile_vstp_parr.second_.first_ = nested_float_col_to_nested_sta(vst_tuple[0])
        tile_vstp_parr.second_.second_.resize(len(vst_tuple[1]))
        for i, (stix, st_vertices) in enumerate(vst_tuple[1].items()):
            sub_tile_vertices_parr = tile_vstp_parr.second_.second_.array_[i]
            sub_tile_vertices_parr.first_ = safe_convert_to_c(tix_t, stix)
            sub_tile_vertices_parr.second_ = nested_float_col_to_nested_sta(st_vertices)
    return gtvpa


def grid_tile_vertices_pair_array_to_dict(
    gtvpa: ctypes.Structure,  # GridTileVerticesPairArray
) -> typing.Dict[
    int,  # Tile index
    typing.Tuple[
        typing.List[typing.List[float]],  # Tile vertices
        typing.Dict[
            int,  # Sub tile index
            typing.List[typing.List[float]],  # Sub tile vertices
        ],
    ],
]:
    if gtvpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(gtvpa.array_)
        for i in range(gtvpa.size_):
            tile_vstp_parr = gtvpa.array_[i]
            tix = c_data_to_py_int(tile_vstp_parr.first_)
            t_vertices = nested_sta_to_nested_float_col(tile_vstp_parr.second_.first_)
            st_dict = dict()
            if tile_vstp_parr.second_.second_.size_ > 0:
                check_ptr(tile_vstp_parr.second_.second_.array_)
                for j in range(tile_vstp_parr.second_.second_.size_):
                    sub_tile_vertices_parr = tile_vstp_parr.second_.second_.array_[j]
                    stix = c_data_to_py_int(sub_tile_vertices_parr.first_)
                    st_dict[stix] = nested_sta_to_nested_float_col(
                        sub_tile_vertices_parr.second_
                    )

                if len(st_dict) != tile_vstp_parr.second_.second_.size_:
                    raise ValueError("Corrupted grid sub tile vertices array")

            res[tix] = (t_vertices, st_dict)

        if len(res) != gtvpa.size_:
            raise ValueError("Corrupted grid tile vertices array")

        return res


def dict_to_timer_data_pair_array(
    d: typing.Dict[str, float],  # Timer name : Time
) -> ctypes.Structure:  # TimerDataPairArray
    tdpa = TimerDataPairArray()
    tdpa.resize(len(d))
    for i, (timer_name, time) in enumerate(d.items()):
        td = tdpa.array_[i]
        td.first_ = str_to_carr(timer_name)
        td.second_ = safe_convert_to_c(ctypes.c_double, time)
    return tdpa


def timer_data_pair_array_to_dict(
    tdpa: ctypes.Structure,  # TimerDataPairArray
) -> typing.Dict[str, float]:  # Timer name : Time
    if tdpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(tdpa.array_)
        for i in range(tdpa.size_):
            td = tdpa.array_[i]
            timer_name = carr_to_str(td.first_)
            timer_time = c_data_to_py_float(td.second_)
            res[timer_name] = timer_time

        if len(res) != tdpa.size_:
            raise ValueError("Corrupted timer data pair array")

        return res


_CONVERTERS = {
    mult_t: (c_data_to_py_int, lambda val: safe_convert_to_c(c_type=mult_t, p_val=val)),
    ctypes.c_bool: (
        c_data_to_py_bool,
        lambda val: safe_convert_to_c(c_type=ctypes.c_bool, p_val=val),
    ),
    CharArray: (carr_to_str, str_to_carr),
    TileIdxArray: (tia_to_int_col, int_col_to_tia),
    SpaceTArray: (sta_to_float_col, float_col_to_sta),
    NestedCharArray: (nested_carr_to_str_col, str_col_to_nested_carr),
    NestedTileIdxArray: (nested_tia_to_nested_int_col, nested_int_col_to_nested_tia),
    NestedSpaceTArray: (nested_sta_to_nested_float_col, nested_float_col_to_nested_sta),
    TiledNodeSequencePairArray: (
        tiled_node_sequence_pair_array_to_dict,
        dict_to_tiled_node_sequence_pair_array,
    ),
    ConnectionPairArray: (connection_pair_array_to_dict, dict_to_connection_pair_array),
    NodeCoordPairArray: (node_coord_pair_array_to_dict, dict_to_node_coord_pair_array),
    TimerDataPairArray: (timer_data_pair_array_to_dict, dict_to_timer_data_pair_array),
}

_RCIS_FIELDS = (
    ("incoming_connections", ConnectionPairArray, lambda: dict()),
    ("outgoing_connections", ConnectionPairArray, lambda: dict()),
)

_MPS_FIELDS = (
    ("mask_blueprint_name", CharArray, lambda: ""),
    ("mask_blueprint_params", SpaceTArray, lambda: tuple()),
    ("source_mask_name", CharArray, lambda: ""),
    ("source_mask_params", SpaceTArray, lambda: tuple()),
    ("target_mask_name", CharArray, lambda: ""),
    ("target_mask_params", SpaceTArray, lambda: tuple()),
)

_CPS_FIELDS = (
    ("edge_wrap", ctypes.c_bool, lambda: False),
    ("only_neighborhood", ctypes.c_bool, lambda: False),
    ("inverted_conn_rule", ctypes.c_bool, lambda: False),
    ("allow_multiplicity", ctypes.c_bool, lambda: False),
    ("allow_self_connections", ctypes.c_bool, lambda: False),
    ("total_number_connections", mult_t, lambda: 0),
    ("conn_gen_name", CharArray, lambda: ""),
    ("weight_df_name", CharArray, lambda: ""),
    ("weight_df_params", SpaceTArray, lambda: tuple()),
    ("weight_ufs_names", NestedCharArray, lambda: tuple()),
    ("weight_ufs_params", NestedSpaceTArray, lambda: tuple()),
    ("delay_df_name", CharArray, lambda: ""),
    ("delay_df_params", SpaceTArray, lambda: tuple()),
    ("delay_ufs_names", NestedCharArray, lambda: tuple()),
    ("delay_ufs_params", NestedSpaceTArray, lambda: tuple()),
    ("prob_df_name", CharArray, lambda: ""),
    ("prob_df_params", SpaceTArray, lambda: tuple()),
    ("prob_ufs_names", NestedCharArray, lambda: tuple()),
    ("prob_ufs_params", NestedSpaceTArray, lambda: tuple()),
)


def io_struct_template(
    field_map,
) -> type[ctypes.Structure]:
    class IOStruct(ctypes.Structure):
        _fields_ = [(field[0], field[1]) for field in field_map]
        _fmp = field_map

        def to_dict(self):
            res = dict()
            for name, type, _ in self._fmp:
                try:
                    res[name] = _CONVERTERS[type][0](self.__getattribute__(name))
                except KeyError:
                    raise KeyError("Unknown field in structure")
            return res

        def from_dict(self, d: dict):
            for name, type, dfg in self._fmp:
                try:
                    if name in d:
                        self.__setattr__(name, _CONVERTERS[type][1](d[name]))
                    else:
                        self.__setattr__(
                            name,
                            _CONVERTERS[type][1](dfg()),
                        )
                except KeyError:
                    raise KeyError("Unknown field in structure")

    return IOStruct


RCIStruct = io_struct_template(_RCIS_FIELDS)
MPStruct = io_struct_template(_MPS_FIELDS)
CPStruct = io_struct_template(_CPS_FIELDS)


_C_SAPI = ctypes.CDLL(str(pathlib.Path(__file__).parent / "libsapi.so"))

_C_SAPI.init.argtypes = (
    vp_t,
    ctypes.POINTER(ctypes.c_char_p),
)
_C_SAPI.init.restype = ctypes.c_bool
_C_SAPI.reset_api.restype = ctypes.c_bool
_C_SAPI.free_gc.restype = ctypes.c_bool
_C_SAPI.get_rank.restype = OptionalIndex
_C_SAPI.get_num_processes.restype = OptionalIndex
_C_SAPI.get_num_threads.restype = OptionalIndex
_C_SAPI.set_num_threads.argtypes = (ctypes.POINTER(vp_t),)
_C_SAPI.set_num_threads.restype = ctypes.c_bool
_C_SAPI.get_rng_seed.restype = OptionalIndex
_C_SAPI.set_rng_seed.argtypes = (ctypes.POINTER(ctypes.c_uint32),)
_C_SAPI.set_rng_seed.restype = ctypes.c_bool
_C_SAPI.get_rng_type.restype = ctypes.POINTER(CharArray)
_C_SAPI.set_rng_type.argtypes = (ctypes.POINTER(CharArray),)
_C_SAPI.set_rng_type.restype = ctypes.c_bool

_C_SAPI.generate_tile_grid.argtypes = (
    ctypes.POINTER(SpaceTArray),
    ctypes.POINTER(TileIdxArray),
    ctypes.POINTER(CharArray),
    ctypes.POINTER(SpaceTArray),
    ctypes.POINTER(NestedTileIdxArray),
    ctypes.POINTER(split_t),
    ctypes.POINTER(ctypes.c_bool),
)
_C_SAPI.generate_tile_grid.restype = ctypes.c_bool

_C_SAPI.generate_nodes_in_grid.argtypes = (
    ctypes.POINTER(lnix_t),
    ctypes.POINTER(TileIdxArray),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
)
_C_SAPI.generate_nodes_in_grid.restype = OptionalIndex

_C_SAPI.insert_positions_in_grid.argtypes = (ctypes.POINTER(NestedSpaceTArray),)
_C_SAPI.insert_positions_in_grid.restype = pair_template(
    OptionalIndex, ctypes.POINTER(NestedSpaceTArray)
)

_C_SAPI.compute_spatial_connections.argtypes = (
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(MPStruct),
    ctypes.POINTER(CPStruct),
)
_C_SAPI.compute_spatial_connections.restype = OptionalIndex

_C_SAPI.get_nodes.argtypes = (
    ctypes.POINTER(OptionalIndex),
    ctypes.POINTER(MPStruct),
)
_C_SAPI.get_nodes.restype = ctypes.POINTER(NestedNodeCoordPairArray)

_C_SAPI.get_distributed_node_sequences.argtypes = (ctypes.POINTER(ctypes.c_size_t),)
_C_SAPI.get_distributed_node_sequences.restype = ctypes.POINTER(
    TiledNodeSequencePairArray
)

_C_SAPI.get_spatial_connections.argtypes = (ctypes.POINTER(ctypes.c_size_t),)
_C_SAPI.get_spatial_connections.restype = ctypes.POINTER(RCIStruct)

_C_SAPI.get_grid_vertices.restype = ctypes.POINTER(GridTileVerticesPairArray)
_C_SAPI.get_timer_data.restype = ctypes.POINTER(TimerDataPairArray)


def check_optional(opt: ctypes.Structure):
    check_bool(opt.first_)
    return int(opt.second_)


def safe_ptr_deref(ptr: ctypes._Pointer):
    check_ptr(ptr)
    return ptr.contents


def init() -> None:
    argc = len(sys.argv)
    c_argc = safe_convert_to_c(ctypes.c_int, argc)
    c_argv = (ctypes.c_char_p * (argc + 1))()
    for i, arg in enumerate(sys.argv):
        c_argv[i] = arg.encode("utf-8", "strict") + b"\0"
    c_argv[argc] = None
    check_bool(_C_SAPI.init(c_argc, c_argv))


def reset_api() -> None:
    check_bool(_C_SAPI.reset_api())


def free_gc() -> None:
    check_bool(_C_SAPI.free_gc())


def get_rank() -> int:
    return check_optional(_C_SAPI.get_rank())


def get_num_processes() -> int:
    return check_optional(_C_SAPI.get_num_processes())


def get_num_threads() -> int:
    return check_optional(_C_SAPI.get_num_threads())


def set_num_threads(num_threads: int) -> None:
    c_vp = safe_convert_to_c(vp_t, num_threads)
    check_bool(_C_SAPI.set_num_threads(ctypes.byref(c_vp)))


def get_rng_seed() -> int:
    return check_optional(_C_SAPI.get_rng_seed())


def set_rng_seed(seed: int) -> None:
    c_seed = safe_convert_to_c(ctypes.c_uint32, seed)
    check_bool(_C_SAPI.set_rng_seed(ctypes.byref(c_seed)))


def get_rng_type() -> str:
    res = carr_to_str(safe_ptr_deref(_C_SAPI.get_rng_type()))
    free_gc()
    return res


def set_rng_type(rng_type: str) -> None:
    carr = str_to_carr(rng_type)
    check_bool(_C_SAPI.set_rng_type(ctypes.byref(carr)))


def generate_tile_grid(
    grid_origin: typing.Collection[float],
    grid_dimensions: typing.Collection[int],
    tile_type: str,
    tile_params: typing.Collection[float],
    rank_tile_owner_ship: typing.Collection[typing.Set[int]],
    splits: int,
    edge_wrap: bool,
) -> None:
    origin_arr = float_col_to_sta(grid_origin)
    dims_arr = int_col_to_tia(grid_dimensions)
    tt_arr = str_to_carr(tile_type)
    tp_arr = float_col_to_sta(tile_params)
    rto_arr = nested_int_col_to_nested_tia(rank_tile_owner_ship)
    c_splits = safe_convert_to_c(split_t, splits)
    c_ew = safe_convert_to_c(ctypes.c_bool, edge_wrap)
    check_bool(
        _C_SAPI.generate_tile_grid(
            ctypes.byref(origin_arr),
            ctypes.byref(dims_arr),
            ctypes.byref(tt_arr),
            ctypes.byref(tp_arr),
            ctypes.byref(rto_arr),
            ctypes.byref(c_splits),
            ctypes.byref(c_ew),
        )
    )


def _parse_distribution_mode(mode: str | int) -> ctypes.c_uint8:
    if isinstance(mode, str):
        match mode.upper():
            case "FREE":
                return ctypes.c_uint8(0)
            case "SQUEEZED":
                return ctypes.c_uint8(1)
            case "BALANCED":
                return ctypes.c_uint8(2)
            case _:
                raise ValueError("Invalid distribution mode")
    if isinstance(mode, int):
        match mode:
            case 0:
                return ctypes.c_uint8(0)
            case 1:
                return ctypes.c_uint8(1)
            case 2:
                return ctypes.c_uint8(2)
            case _:
                raise ValueError("Invalid distribution mode")
    raise TypeError("Invalid distribution mode argument")


def generate_nodes_in_grid(
    num_nodes: int,
    tiles: typing.Set[int] | None = None,
    grid_distribution_mode: str | int = "balanced",
    tile_distribution_mode: str | int = "squeezed",
) -> int:
    c_nn = safe_convert_to_c(lnix_t, num_nodes)
    tiles_arr = TileIdxArray()
    tiles_arr.size_ = 0
    if tiles is not None and len(tiles) > 0:
        tiles_arr = int_col_to_tia(tiles)
    grid_mode = _parse_distribution_mode(grid_distribution_mode)
    tile_mode = _parse_distribution_mode(tile_distribution_mode)
    return check_optional(
        _C_SAPI.generate_nodes_in_grid(
            ctypes.byref(c_nn),
            ctypes.byref(tiles_arr),
            ctypes.byref(grid_mode),
            ctypes.byref(tile_mode),
        )
    )


def insert_positions_in_grid(
    positions: typing.Collection[typing.Collection[float]],
) -> typing.Tuple[int, typing.List[typing.List[float]]]:
    if positions is None or len(positions) < 1:
        raise ValueError("Cannot insert empty position collection")
    c_pos = nested_float_col_to_nested_sta(positions)  # copy 1
    res_pair = _C_SAPI.insert_positions_in_grid(
        ctypes.byref(c_pos)
    )  # internal cpp copy 2 + C leftovers
    del c_pos  # delete copy 1
    index = check_optional(res_pair.first_)
    leftovers = nested_sta_to_nested_float_col(safe_ptr_deref(res_pair.second_))
    free_gc()  # clean C leftover positions
    return index, leftovers


def compute_spatial_connections(
    source_dist_tns_idx: int,
    target_dist_tns_idx: int,
    mask_params: dict,
    conn_params: dict,
) -> int:
    c_idx0 = safe_convert_to_c(ctypes.c_size_t, source_dist_tns_idx)
    c_idx1 = safe_convert_to_c(ctypes.c_size_t, target_dist_tns_idx)
    mps = MPStruct()
    mps.from_dict(mask_params)
    cps = CPStruct()
    cps.from_dict(conn_params)
    return check_optional(
        _C_SAPI.compute_spatial_connections(
            ctypes.byref(c_idx0),
            ctypes.byref(c_idx1),
            ctypes.byref(mps),
            ctypes.byref(cps),
        )
    )


def get_nodes(
    dist_tns_index: int | None = None, mask_params: dict | None = None
) -> typing.Dict[
    int,  # Tile index
    typing.Dict[
        int,  # Leaf index
        typing.Dict[int, typing.List[float]],  # Node index : Coordinates
    ],
]:
    c_opt = OptionalIndex()
    if dist_tns_index is not None:
        c_opt.first_ = safe_convert_to_c(ctypes.c_bool, True)
        c_opt.second_ = safe_convert_to_c(ctypes.c_size_t, dist_tns_index)
    mps = MPStruct()
    if mask_params is not None:
        mps.from_dict(mask_params)
    res = nested_node_coord_pair_array_to_dict(
        safe_ptr_deref(_C_SAPI.get_nodes(ctypes.byref(c_opt), ctypes.byref(mps)))
    )
    free_gc()
    return res


def get_distributed_node_sequences(
    dist_tns_index: int,
) -> typing.Dict[int, typing.Dict[int, typing.Tuple[int, int]]]:
    c_idx = safe_convert_to_c(ctypes.c_size_t, dist_tns_index)
    res = tiled_node_sequence_pair_array_to_dict(
        safe_ptr_deref(_C_SAPI.get_distributed_node_sequences(ctypes.byref(c_idx)))
    )
    free_gc()
    return res


def get_spatial_connections(conn_index: int) -> typing.Tuple[
    typing.Dict[
        int, typing.Dict[int, typing.Dict[int, typing.Tuple[float, float, int]]]
    ],
    typing.Dict[
        int, typing.Dict[int, typing.Dict[int, typing.Tuple[float, float, int]]]
    ],
]:
    c_idx = safe_convert_to_c(ctypes.c_size_t, conn_index)
    res = safe_ptr_deref(
        _C_SAPI.get_spatial_connections(
            ctypes.byref(c_idx),
        )
    ).to_dict()
    free_gc()
    return res["incoming_connections"], res["outgoing_connections"]


def get_grid_vertices() -> typing.Dict[
    int,  # Tile index
    typing.Tuple[
        typing.List[typing.List[float]],  # Tile vertices
        typing.Dict[
            int,  # Sub tile index
            typing.List[typing.List[float]],  # Sub tile vertices
        ],
    ],
]:
    res = grid_tile_vertices_pair_array_to_dict(
        safe_ptr_deref(_C_SAPI.get_grid_vertices())
    )
    free_gc()
    return res


def get_timer_data() -> typing.Dict[str, float]:
    res = timer_data_pair_array_to_dict(safe_ptr_deref(_C_SAPI.get_timer_data()))
    free_gc()
    return res
