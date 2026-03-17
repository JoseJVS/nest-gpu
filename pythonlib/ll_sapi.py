#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CTypes bindings for low level Spatial API.

Authors: JoseJVS.

"""

import ctypes
import typing

vp_t: typing.TypeAlias = ctypes.c_int32
tix_t: typing.TypeAlias = ctypes.c_int32
nix_t: typing.TypeAlias = ctypes.c_int32
lnix_t: typing.TypeAlias = ctypes.c_int64
mult_t: typing.TypeAlias = ctypes.c_uint16
split_t: typing.TypeAlias = ctypes.c_uint8
space_t: typing.TypeAlias = ctypes.c_double
angle_t: typing.TypeAlias = ctypes.c_int16
conn_index_t: typing.TypeAlias = ctypes.c_uint32
conn_param_t: typing.TypeAlias = ctypes.c_float
CData: typing.TypeAlias = ctypes._SimpleCData | ctypes.Structure | ctypes._Pointer


class SpatialNodeSeq:
    def __init__(
        self,
        spatial_index: int,
        total_length: int,
        local_index: int | None = None,
        local_length: int | None = None,
    ) -> None:
        self._spatial_index = spatial_index
        self._total_length = total_length
        self._local_index = local_index
        self._local_length = local_length

    @property
    def spatial_index(self) -> int:
        return self._spatial_index

    @spatial_index.setter
    def spatial_index(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set spatial index")

    @property
    def total_length(self) -> int:
        return self._total_length

    @total_length.setter
    def total_length(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set total length")

    @property
    def local_index(self) -> int | None:
        return self._local_index

    @local_index.setter
    def local_index(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set local index")

    @property
    def local_length(self) -> int | None:
        return self._local_length

    @local_length.setter
    def local_length(self, **_) -> typing.NoReturn:
        raise AttributeError("Cannot set local length")


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
SpatialNodeSequence = triplet_template(ctypes.c_size_t, nix_t, nix_t)
CharArray = array_template(ctypes.c_char)
SpaceTArray = array_template(space_t)
TileIdxArray = array_template(tix_t)
AngleTArray = array_template(angle_t)

NestedCharArray = array_template(CharArray)
NestedSpaceTArray = array_template(SpaceTArray)
NestedTileIdxArray = array_template(TileIdxArray)

TiledNodeSequencePairArray = pair_array_template(
    vp_t, pair_array_template(tix_t, pair_template(nix_t, nix_t))
)
NodeCoordPairArray = pair_array_template(
    tix_t, pair_array_template(nix_t, array_template(space_t))
)
NestedNodeCoordPairArray = pair_array_template(tix_t, NodeCoordPairArray)
GridTileVerticesPairArray = pair_array_template(
    tix_t,
    pair_template(NestedSpaceTArray, pair_array_template(tix_t, NestedSpaceTArray)),
)

DoubleArray = array_template(ctypes.c_double)
RankTimerDataPairArray = pair_array_template(CharArray, ctypes.c_double)
ThreadTimerDataPairArray = pair_array_template(CharArray, DoubleArray)
RecordedTimesArrayPair = pair_template(RankTimerDataPairArray, ThreadTimerDataPairArray)


class ConnectionInfoStruct(ctypes.Structure):
    _fields_ = [
        ("source_index_", conn_index_t),
        ("target_index_", conn_index_t),
        ("connection_weight_", conn_param_t),
        ("connection_delay_", conn_param_t),
    ]


ConnectionInfoPartition = array_template(ConnectionInfoStruct)
ConnectionInfoPairArray = pair_array_template(
    vp_t,
    array_template(ConnectionInfoPartition),
)
RemoteConnectionInfoPair = pair_template(
    ConnectionInfoPairArray, ConnectionInfoPairArray
)


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

int_col_to_ata = lambda col: num_col_to_num_arr(num_arr_type=AngleTArray, num_col=col)
ata_to_int_col = lambda arr: num_arr_to_num_col(num_type=int, num_arr=arr)

float_col_to_da = lambda col: num_col_to_num_arr(num_arr_type=DoubleArray, num_col=col)
da_to_float_col = lambda arr: num_arr_to_num_col(num_type=float, num_arr=arr)

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


def tuple_to_connection_info_struct(
    tup: typing.Tuple[int, int, float, float],
) -> ConnectionInfoStruct:
    if len(tup) != 4:
        raise ValueError("Incorrect tuple length for CIStruct")
    cis = ConnectionInfoStruct()
    cis.source_index_ = safe_convert_to_c(conn_index_t, tup[0])
    cis.target_index_ = safe_convert_to_c(conn_index_t, tup[1])
    cis.connection_weight_ = safe_convert_to_c(conn_param_t, tup[2])
    cis.connection_delay_ = safe_convert_to_c(conn_param_t, tup[3])
    return cis


def connection_info_struct_to_tuple(cis: ConnectionInfoStruct) -> tuple:
    tup = (
        c_data_to_py_int(cis.source_index_),
        c_data_to_py_int(cis.target_index_),
        c_data_to_py_float(cis.connection_weight_),
        c_data_to_py_float(cis.connection_delay_),
    )
    return tup


def list_to_connection_info_partition(
    l: typing.List[typing.Tuple[int, int, float, float]],
) -> ctypes.Structure:  # ConnectionInfoPartition
    cip = ConnectionInfoPartition()
    cip.resize(len(l))
    for i, tup in enumerate(l):
        cip.array_[i] = tuple_to_connection_info_struct(tup)
    return cip


def connection_info_partition_to_list(
    cip: ctypes.Structure,  # ConnectionInfoPartition
) -> typing.List[typing.Tuple[int, int, float, float]]:
    if cip.size_ < 1:
        return []
    else:
        res = []
        check_ptr(cip.array_)
        for i in range(cip.size_):
            res.append(connection_info_struct_to_tuple(cip.array_[i]))
        return res


def dict_to_connection_info_pair_array(
    d: typing.Dict[int, typing.List[typing.List[typing.Tuple[int, int, float, float]]]],
) -> ctypes.Structure:  # ConnectionInfoArray
    cnnpa = ConnectionInfoPairArray()
    cnnpa.resize(len(d))
    for r, (rank, conn_collection) in enumerate(d.items()):
        rank_conn_parr = cnnpa.array_[r]
        rank_conn_parr.first_ = safe_convert_to_c(vp_t, rank)
        rank_conn_parr.second_.resize(len(conn_collection))
        for conn_idx, conn_partition in enumerate(conn_collection):
            rank_conn_parr.second_.array_[conn_idx] = list_to_connection_info_partition(
                conn_partition
            )

    return cnnpa


def connection_info_pair_to_dict(
    cnnpa: ctypes.Structure,  # ConnectionInfoArray
) -> typing.Dict[int, typing.List[typing.List[typing.Tuple[int, int, float, float]]]]:
    if cnnpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(cnnpa.array_)
        for r in range(cnnpa.size_):
            rank_conn_parr = cnnpa.array_[r]
            rank = c_data_to_py_int(rank_conn_parr.first_)
            conn_list = res[rank] = list()
            if rank_conn_parr.second_.size_ > 0:
                check_ptr(rank_conn_parr.second_.array_)
                for partition_index in range(rank_conn_parr.second_.size_):
                    conn_list.append(
                        connection_info_partition_to_list(
                            rank_conn_parr.second_.array_[partition_index]
                        )
                    )
                if len(conn_list) != rank_conn_parr.second_.size_:
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


def dict_to_rank_timer_data_pair_array(
    d: typing.Dict[str, float],  # Timer name : Time
) -> ctypes.Structure:  # RankTimerDataPairArray
    rtdpa = RankTimerDataPairArray()
    rtdpa.resize(len(d))
    for i, (timer_name, time) in enumerate(d.items()):
        td = rtdpa.array_[i]
        td.first_ = str_to_carr(timer_name)
        td.second_ = safe_convert_to_c(ctypes.c_double, time)
    return rtdpa


def rank_timer_data_pair_array_to_dict(
    rtdpa: ctypes.Structure,  # RankTimerDataPairArray
) -> typing.Dict[str, float]:  # Timer name : Time
    if rtdpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(rtdpa.array_)
        for i in range(rtdpa.size_):
            td = rtdpa.array_[i]
            timer_name = carr_to_str(td.first_)
            timer_time = c_data_to_py_float(td.second_)
            res[timer_name] = timer_time

        if len(res) != rtdpa.size_:
            raise ValueError("Corrupted rank timer data pair array")

        return res


def dict_to_thread_timer_data_pair_array(
    d: typing.Dict[str, typing.List[float]],  # Timer name : Time
) -> ctypes.Structure:  # TimerDataPairArray
    ttdpa = ThreadTimerDataPairArray()
    ttdpa.resize(len(d))
    for i, (timer_name, times) in enumerate(d.items()):
        td = ttdpa.array_[i]
        td.first_ = str_to_carr(timer_name)
        td.second_ = float_col_to_da(times)
    return ttdpa


def thread_timer_data_pair_array_to_dict(
    rtdpa: ctypes.Structure,  # TimerDataPairArray
) -> typing.Dict[str, float]:  # Timer name : Time
    if rtdpa.size_ < 1:
        return dict()
    else:
        res = dict()
        check_ptr(rtdpa.array_)
        for i in range(rtdpa.size_):
            td = rtdpa.array_[i]
            timer_name = carr_to_str(td.first_)
            timer_time = da_to_float_col(td.second_)
            res[timer_name] = timer_time

        if len(res) != rtdpa.size_:
            raise ValueError("Corrupted rank timer data pair array")

        return res


_CONVERTERS = {
    mult_t: (c_data_to_py_int, lambda val: safe_convert_to_c(c_type=mult_t, p_val=val)),
    ctypes.c_bool: (
        c_data_to_py_bool,
        lambda val: safe_convert_to_c(c_type=ctypes.c_bool, p_val=val),
    ),
    CharArray: (carr_to_str, str_to_carr),
    NestedCharArray: (nested_carr_to_str_col, str_col_to_nested_carr),
    SpaceTArray: (sta_to_float_col, float_col_to_sta),
    NestedSpaceTArray: (nested_sta_to_nested_float_col, nested_float_col_to_nested_sta),
}

_MPS_FIELDS = (
    ("mask_blueprint_name", CharArray, lambda: ""),
    ("mask_blueprint_params", SpaceTArray, lambda: tuple()),
    ("mask_blueprint_offset", SpaceTArray, lambda: tuple()),
    ("source_mask_name", CharArray, lambda: ""),
    ("source_mask_origin", SpaceTArray, lambda: tuple()),
    ("source_mask_params", SpaceTArray, lambda: tuple()),
    ("source_mask_offset", SpaceTArray, lambda: tuple()),
    ("target_mask_name", CharArray, lambda: ""),
    ("target_mask_origin", SpaceTArray, lambda: tuple()),
    ("target_mask_params", SpaceTArray, lambda: tuple()),
    ("target_mask_offset", SpaceTArray, lambda: tuple()),
)

_CPS_FIELDS = (
    ("edge_wrap", ctypes.c_bool, lambda: False),
    ("only_neighborhood", ctypes.c_bool, lambda: False),
    ("inverted_conn_rule", ctypes.c_bool, lambda: False),
    ("allow_multiplicity", ctypes.c_bool, lambda: False),
    ("allow_self_connections", ctypes.c_bool, lambda: False),
    ("partition_connections_by_source", ctypes.c_bool, lambda: False),
    ("connection_counts", mult_t, lambda: 0),
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


MPStruct = io_struct_template(_MPS_FIELDS)
CPStruct = io_struct_template(_CPS_FIELDS)


def check_optional(opt: ctypes.Structure):
    check_bool(opt.first_)
    return int(opt.second_)


def safe_ptr_deref(ptr: ctypes._Pointer):
    check_ptr(ptr)
    return ptr.contents


def parse_distribution_mode(mode: str | int) -> ctypes.c_uint8:
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
