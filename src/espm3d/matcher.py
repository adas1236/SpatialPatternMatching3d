"""ESPM-3D: spatial pattern matching with an inverted octree index.

This module generalizes the 2-D ESPM idea to 3-D:
  * one octree per keyword, like an inverted quadtree/octree;
  * node-level pruning with minimum bounding rectangular prisms (MBRPs);
  * object-level edge matching; and
  * a final join over edge matches.

The code uses only the Python standard library.  It favors correctness and
readability while still avoiding the most expensive brute-force scans where the
index can safely prune candidates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import count
from math import inf
from typing import (
    DefaultDict,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

ObjectId = Hashable
VertexId = Hashable
Point3D = Tuple[float, float, float]
EPS = 1e-12


class EdgeSign(Enum):
    """Meaning of the lower bound on a pattern edge.

    For an edge (source, target) with distance interval [lower, upper]:

    INCLUSION ("--")
        Only the source-target object pair must be in [lower, upper].  Other
        same-keyword objects closer than lower are allowed.

    SOURCE_EXCLUDES_TARGET ("->")
        The pair must be in [lower, upper], and no target-keyword object may be
        closer than lower to the source object.

    TARGET_EXCLUDES_SOURCE ("<-")
        The pair must be in [lower, upper], and no source-keyword object may be
        closer than lower to the target object.

    MUTUAL_EXCLUSION ("<->")
        Both exclusion constraints hold.
    """

    INCLUSION = "--"
    SOURCE_EXCLUDES_TARGET = "->"
    TARGET_EXCLUDES_SOURCE = "<-"
    MUTUAL_EXCLUSION = "<->"

    @classmethod
    def parse(cls, value: "EdgeSign | str") -> "EdgeSign":
        if isinstance(value, EdgeSign):
            return value
        aliases = {
            "--": cls.INCLUSION,
            "include": cls.INCLUSION,
            "inclusion": cls.INCLUSION,
            "mutual_inclusion": cls.INCLUSION,
            "->": cls.SOURCE_EXCLUDES_TARGET,
            "source_excludes_target": cls.SOURCE_EXCLUDES_TARGET,
            "source_excludes": cls.SOURCE_EXCLUDES_TARGET,
            "<-": cls.TARGET_EXCLUDES_SOURCE,
            "target_excludes_source": cls.TARGET_EXCLUDES_SOURCE,
            "target_excludes": cls.TARGET_EXCLUDES_SOURCE,
            "<->": cls.MUTUAL_EXCLUSION,
            "$": cls.MUTUAL_EXCLUSION,
            "mutual_exclusion": cls.MUTUAL_EXCLUSION,
        }
        key = value.strip().lower()
        if key not in aliases:
            raise ValueError(f"unknown edge sign: {value!r}")
        return aliases[key]

    @property
    def is_exclusive(self) -> bool:
        return self is not EdgeSign.INCLUSION


@dataclass(frozen=True)
class SpatialObject:
    """A 3-D spatio-textual object."""

    id: ObjectId
    x: float
    y: float
    z: float
    keywords: FrozenSet[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "z", float(self.z))
        object.__setattr__(self, "keywords", frozenset(self.keywords))
        if not self.keywords:
            raise ValueError(f"object {self.id!r} has no keywords")

    @property
    def point(self) -> Point3D:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class PatternEdge:
    """A spatial constraint between two pattern vertices."""

    source: VertexId
    target: VertexId
    lower: float
    upper: float
    sign: EdgeSign | str = EdgeSign.INCLUSION

    def __post_init__(self) -> None:
        lower = float(self.lower)
        upper = float(self.upper)
        if lower < 0 or upper < 0:
            raise ValueError("distance bounds must be non-negative")
        if lower > upper:
            raise ValueError("lower distance bound cannot exceed upper bound")
        if self.source == self.target:
            raise ValueError("self-edges are not supported")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "sign", EdgeSign.parse(self.sign))


@dataclass(frozen=True)
class SpatialPattern:
    """A graph-shaped 3-D spatial pattern.

    vertices maps pattern vertex ids to required object keywords.
    """

    vertices: Mapping[VertexId, str]
    edges: Sequence[PatternEdge]

    def __post_init__(self) -> None:
        vertices = dict(self.vertices)
        if not vertices:
            raise ValueError("pattern must contain at least one vertex")
        for v, keyword in vertices.items():
            if not isinstance(keyword, str) or not keyword:
                raise ValueError(f"vertex {v!r} has an invalid keyword {keyword!r}")
        normalized_edges = tuple(self.edges)
        for edge in normalized_edges:
            if edge.source not in vertices or edge.target not in vertices:
                raise ValueError(f"edge {edge!r} references an unknown vertex")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "edges", normalized_edges)


@dataclass(frozen=True)
class Box3D:
    """Axis-aligned rectangular prism used as an MBRP."""

    mins: Point3D
    maxs: Point3D

    def __post_init__(self) -> None:
        mins = tuple(float(v) for v in self.mins)
        maxs = tuple(float(v) for v in self.maxs)
        if len(mins) != 3 or len(maxs) != 3:
            raise ValueError("Box3D requires 3-D min and max points")
        if any(lo > hi for lo, hi in zip(mins, maxs)):
            raise ValueError(f"invalid box: mins={mins}, maxs={maxs}")
        object.__setattr__(self, "mins", mins)  # type: ignore[arg-type]
        object.__setattr__(self, "maxs", maxs)  # type: ignore[arg-type]

    @classmethod
    def from_points(cls, points: Iterable[Point3D]) -> "Box3D":
        points = list(points)
        if not points:
            raise ValueError("cannot build a box from no points")
        mins = tuple(min(p[d] for p in points) for d in range(3))
        maxs = tuple(max(p[d] for p in points) for d in range(3))
        return cls(mins, maxs)  # type: ignore[arg-type]

    @property
    def midpoint(self) -> Point3D:
        return tuple((self.mins[d] + self.maxs[d]) * 0.5 for d in range(3))  # type: ignore[return-value]

    def expanded_to_nonzero(self, min_padding: float = 1.0) -> "Box3D":
        """Return a box with non-zero width in every dimension.

        Degenerate dimensions are expanded.  This is for octree cell splitting;
        MBRPs still use the true object bounds.
        """

        extents = [self.maxs[d] - self.mins[d] for d in range(3)]
        scale = max(max(extents), min_padding)
        pad = max(scale * 1e-9, 1e-9)
        mins = list(self.mins)
        maxs = list(self.maxs)
        for d in range(3):
            if maxs[d] - mins[d] <= 0:
                mins[d] -= pad
                maxs[d] += pad
            else:
                # Include points on the root boundary robustly.
                mins[d] -= pad
                maxs[d] += pad
        return Box3D(tuple(mins), tuple(maxs))  # type: ignore[arg-type]

    def child_cell(self, index: int) -> "Box3D":
        """Return one of the 8 octants of this cell.

        index bits: x=1, y=2, z=4 choose the upper half in that dimension.
        """

        mid = self.midpoint
        mins = list(self.mins)
        maxs = list(self.maxs)
        for d, bit in enumerate((1, 2, 4)):
            if index & bit:
                mins[d] = mid[d]
            else:
                maxs[d] = mid[d]
        return Box3D(tuple(mins), tuple(maxs))  # type: ignore[arg-type]


def squared_distance_points(a: Point3D, b: Point3D) -> float:
    return sum((a[d] - b[d]) ** 2 for d in range(3))


def min_distance_sq_point_box(point: Point3D, box: Box3D) -> float:
    """Squared minimum Euclidean distance from a point to an MBRP."""

    total = 0.0
    for d in range(3):
        if point[d] < box.mins[d]:
            total += (box.mins[d] - point[d]) ** 2
        elif point[d] > box.maxs[d]:
            total += (point[d] - box.maxs[d]) ** 2
    return total


def min_distance_sq_boxes(a: Box3D, b: Box3D) -> float:
    """Squared minimum Euclidean distance between two MBRPs."""

    total = 0.0
    for d in range(3):
        if a.maxs[d] < b.mins[d]:
            total += (b.mins[d] - a.maxs[d]) ** 2
        elif b.maxs[d] < a.mins[d]:
            total += (a.mins[d] - b.maxs[d]) ** 2
    return total


def max_distance_sq_boxes(a: Box3D, b: Box3D) -> float:
    """Squared maximum Euclidean distance between two MBRPs."""

    total = 0.0
    for d in range(3):
        total += max((a.mins[d] - b.maxs[d]) ** 2, (a.maxs[d] - b.mins[d]) ** 2)
    return total


class OctreeNode:
    """A non-empty octree node.

    cell is the regular octree cell; mbr is the tight MBRP of contained objects.
    """

    __slots__ = ("cell", "mbr", "level", "objects", "children", "serial")
    _serials = count()

    def __init__(
        self,
        *,
        cell: Box3D,
        mbr: Box3D,
        level: int,
        objects: Sequence[SpatialObject],
    ) -> None:
        self.cell = cell
        self.mbr = mbr
        self.level = level
        self.objects = tuple(objects)
        self.children: Tuple[OctreeNode, ...] = ()
        self.serial = next(OctreeNode._serials)

    def __repr__(self) -> str:
        return f"OctreeNode(level={self.level}, n={len(self.objects)}, id={self.serial})"


class Octree:
    """A compact octree over all objects containing one keyword."""

    def __init__(
        self,
        objects: Sequence[SpatialObject],
        *,
        root_cell: Box3D,
        capacity: int = 64,
        min_level: int = 0,
        max_level: int = 12,
    ) -> None:
        if not objects:
            raise ValueError("Octree requires at least one object")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if min_level < 0 or max_level < 0 or min_level > max_level:
            raise ValueError("expected 0 <= min_level <= max_level")
        self.capacity = capacity
        self.min_level = min_level
        self.max_level = max_level
        self.root = self._build(tuple(objects), root_cell, 0)
        self._level_cache: Dict[int, Tuple[OctreeNode, ...]] = {0: (self.root,)}

    def _build(self, objects: Tuple[SpatialObject, ...], cell: Box3D, level: int) -> OctreeNode:
        node = OctreeNode(
            cell=cell,
            mbr=Box3D.from_points(obj.point for obj in objects),
            level=level,
            objects=objects,
        )
        should_split = level < self.max_level and (
            level < self.min_level or len(objects) > self.capacity
        )
        if not should_split:
            return node

        mid = cell.midpoint
        buckets: List[List[SpatialObject]] = [[] for _ in range(8)]
        for obj in objects:
            idx = 0
            if obj.x >= mid[0]:
                idx |= 1
            if obj.y >= mid[1]:
                idx |= 2
            if obj.z >= mid[2]:
                idx |= 4
            buckets[idx].append(obj)

        children: List[OctreeNode] = []
        for idx, bucket in enumerate(buckets):
            if bucket:
                children.append(self._build(tuple(bucket), cell.child_cell(idx), level + 1))
        node.children = tuple(children)
        return node

    @staticmethod
    def children_or_self(node: OctreeNode) -> Tuple[OctreeNode, ...]:
        # Virtual chains for short branches: a leaf repeats at deeper levels.
        return node.children if node.children else (node,)

    def nodes_at_level(self, level: int) -> Tuple[OctreeNode, ...]:
        """Return non-empty nodes at a level, using virtual leaves as needed."""

        if level < 0:
            raise ValueError("level must be non-negative")
        if level in self._level_cache:
            return self._level_cache[level]

        out: List[OctreeNode] = []

        def visit(node: OctreeNode) -> None:
            if node.level >= level or not node.children:
                out.append(node)
                return
            for child in node.children:
                visit(child)

        visit(self.root)
        self._level_cache[level] = tuple(out)
        return self._level_cache[level]

    def any_point_within_radius(self, point: Point3D, radius: float) -> bool:
        """Return True if any indexed object lies at strict distance < radius."""

        if radius <= 0:
            return False
        threshold = radius * radius

        def visit(node: OctreeNode) -> bool:
            if min_distance_sq_point_box(point, node.mbr) >= threshold - EPS:
                return False
            if node.children:
                return any(visit(child) for child in node.children)
            return any(squared_distance_points(point, obj.point) < threshold - EPS for obj in node.objects)

        return visit(self.root)


class InvertedOctreeIndex:
    """One octree per keyword."""

    def __init__(
        self,
        objects: Iterable[SpatialObject],
        *,
        capacity: int = 64,
        min_level: int = 0,
        max_level: int = 12,
    ) -> None:
        self.objects: Dict[ObjectId, SpatialObject] = {}
        for obj in objects:
            if obj.id in self.objects:
                raise ValueError(f"duplicate object id: {obj.id!r}")
            self.objects[obj.id] = obj
        if not self.objects:
            raise ValueError("index requires at least one object")

        self.capacity = capacity
        self.min_level = min_level
        self.max_level = max_level
        global_mbr = Box3D.from_points(obj.point for obj in self.objects.values())
        self.root_cell = global_mbr.expanded_to_nonzero()

        by_keyword: DefaultDict[str, List[SpatialObject]] = defaultdict(list)
        for obj in self.objects.values():
            for keyword in obj.keywords:
                by_keyword[keyword].append(obj)
        self.by_keyword: Dict[str, Tuple[SpatialObject, ...]] = {
            keyword: tuple(items) for keyword, items in by_keyword.items()
        }
        self.trees: Dict[str, Octree] = {
            keyword: Octree(
                items,
                root_cell=self.root_cell,
                capacity=capacity,
                min_level=min_level,
                max_level=max_level,
            )
            for keyword, items in self.by_keyword.items()
        }

    def objects_for_keyword(self, keyword: str) -> Tuple[SpatialObject, ...]:
        return self.by_keyword.get(keyword, ())

    def tree_for_keyword(self, keyword: str) -> Optional[Octree]:
        return self.trees.get(keyword)

    def object(self, object_id: ObjectId) -> SpatialObject:
        return self.objects[object_id]


class DisjointSet:
    def __init__(self, items: Iterable[VertexId]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: VertexId) -> VertexId:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: VertexId, b: VertexId) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def connected(self, a: VertexId, b: VertexId) -> bool:
        return self.find(a) == self.find(b)


@dataclass
class EdgeMatchStore:
    pairs: Tuple[Tuple[ObjectId, ObjectId], ...]
    pair_set: Set[Tuple[ObjectId, ObjectId]]
    by_source: Dict[ObjectId, Tuple[ObjectId, ...]]
    by_target: Dict[ObjectId, Tuple[ObjectId, ...]]

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[ObjectId, ObjectId]]) -> "EdgeMatchStore":
        unique = tuple(dict.fromkeys(pairs))
        by_source: DefaultDict[ObjectId, List[ObjectId]] = defaultdict(list)
        by_target: DefaultDict[ObjectId, List[ObjectId]] = defaultdict(list)
        for src, tgt in unique:
            by_source[src].append(tgt)
            by_target[tgt].append(src)
        return cls(
            pairs=unique,
            pair_set=set(unique),
            by_source={k: tuple(v) for k, v in by_source.items()},
            by_target={k: tuple(v) for k, v in by_target.items()},
        )

    def __len__(self) -> int:
        return len(self.pairs)


@dataclass
class MatchStats:
    nmatch_counts_by_level: List[Dict[int, int]]
    ematch_counts: Dict[int, int]
    skip_edges: Set[int]


class ESPM3DMatcher:
    """Generalized ESPM matcher for 3-D objects.

    Parameters
    ----------
    index:
        Inverted octree index over the data set.
    require_distinct_objects:
        If False, a single multi-keyword object may satisfy multiple pattern
        vertices when all edge constraints allow it.  This follows the paper's
        surjective mapping definition.  Set True when pattern vertices must map
        to different object ids.
    """

    def __init__(self, index: InvertedOctreeIndex, *, require_distinct_objects: bool = False) -> None:
        self.index = index
        self.require_distinct_objects = require_distinct_objects
        self.last_stats: Optional[MatchStats] = None

    def match(self, pattern: SpatialPattern, *, limit: Optional[int] = None) -> List[Dict[VertexId, ObjectId]]:
        """Return all matches as dictionaries {pattern_vertex: object_id}.

        If limit is not None, the search stops after producing that many partial
        final matches.  Leave it as None to enumerate all matches.
        """

        if limit is not None and limit < 1:
            return []
        self.last_stats = None

        # Every pattern keyword must be present in the inverted index.
        for keyword in pattern.vertices.values():
            if self.index.tree_for_keyword(keyword) is None:
                self.last_stats = MatchStats([], {}, set())
                return []

        if not pattern.edges:
            matches = self._extend_isolated_vertices([{}], pattern, limit)
            self.last_stats = MatchStats([], {}, set())
            return matches

        final_nmatches, n_counts_by_level = self._compute_nmatches(pattern)
        if final_nmatches is None:
            self.last_stats = MatchStats(n_counts_by_level, {}, set())
            return []

        edge_ids = list(range(len(pattern.edges)))
        n_counts = {eid: len(final_nmatches[eid]) for eid in edge_ids}
        ematch_order = self._edge_order(pattern, edge_ids, n_counts)
        skip_edges = self._identify_skip_edges(pattern, ematch_order)

        ematches: Dict[int, EdgeMatchStore] = {}
        for eid in ematch_order:
            if eid in skip_edges:
                continue
            store = self._compute_ematches_for_edge(pattern, eid, final_nmatches[eid])
            if len(store) == 0:
                self.last_stats = MatchStats(n_counts_by_level, {eid: 0}, skip_edges)
                return []
            ematches[eid] = store

        join_order = self._join_order(edge_ids, ematches, n_counts, skip_edges)
        assignments: List[Dict[VertexId, ObjectId]] = [{}]
        for eid in join_order:
            edge = pattern.edges[eid]
            # Do not apply the public result limit to intermediate join tuples:
            # later edges, especially skip-edges, can filter early tuples.
            # Truncating before all constraints are checked may drop all valid
            # complete matches even when they exist.
            if eid in skip_edges:
                assignments = self._filter_by_skip_edge(assignments, edge, None)
            else:
                assignments = self._extend_with_edge(assignments, edge, ematches[eid], None)
            if not assignments:
                break

        if assignments:
            assignments = self._extend_isolated_vertices(assignments, pattern, limit)
        elif limit is not None:
            assignments = []

        if limit is not None and len(assignments) > limit:
            assignments = assignments[:limit]

        self.last_stats = MatchStats(
            nmatch_counts_by_level=n_counts_by_level,
            ematch_counts={eid: len(store) for eid, store in ematches.items()},
            skip_edges=skip_edges,
        )
        return assignments

    # ---------- Node-level matching ----------

    def _compute_nmatches(
        self, pattern: SpatialPattern
    ) -> Tuple[Optional[Dict[int, Tuple[Tuple[OctreeNode, OctreeNode], ...]]], List[Dict[int, int]]]:
        edges = pattern.edges
        edge_ids = list(range(len(edges)))
        current: Dict[int, Tuple[Tuple[OctreeNode, OctreeNode], ...]] = {}
        counts_by_level: List[Dict[int, int]] = []

        # Level 0: each keyword tree contributes its root node.
        for eid, edge in enumerate(edges):
            src_tree = self.index.tree_for_keyword(pattern.vertices[edge.source])
            tgt_tree = self.index.tree_for_keyword(pattern.vertices[edge.target])
            assert src_tree is not None and tgt_tree is not None
            pair = (src_tree.root, tgt_tree.root)
            if self._node_pair_is_nmatch(pattern, eid, pair[0], pair[1], level=0):
                current[eid] = (pair,)
            else:
                return None, counts_by_level
        counts_by_level.append({eid: len(current[eid]) for eid in edge_ids})

        # Refine level by level.  Short branches are treated as virtual chains.
        for level in range(1, self.index.max_level + 1):
            previous_counts = {eid: len(current[eid]) for eid in edge_ids}
            order = self._edge_order(pattern, edge_ids, previous_counts)
            next_matches: Dict[int, Tuple[Tuple[OctreeNode, OctreeNode], ...]] = {}
            allowed: Dict[VertexId, Optional[Set[OctreeNode]]] = {
                vertex: None for vertex in pattern.vertices
            }

            for eid in order:
                edge = edges[eid]
                src_allowed = allowed[edge.source]
                tgt_allowed = allowed[edge.target]
                pairs: List[Tuple[OctreeNode, OctreeNode]] = []
                seen: Set[Tuple[OctreeNode, OctreeNode]] = set()

                for parent_src, parent_tgt in current[eid]:
                    src_children = Octree.children_or_self(parent_src)
                    tgt_children = Octree.children_or_self(parent_tgt)
                    for src_node in src_children:
                        if src_allowed is not None and src_node not in src_allowed:
                            continue
                        for tgt_node in tgt_children:
                            if tgt_allowed is not None and tgt_node not in tgt_allowed:
                                continue
                            node_pair = (src_node, tgt_node)
                            if node_pair in seen:
                                continue
                            if self._node_pair_is_nmatch(pattern, eid, src_node, tgt_node, level=level):
                                seen.add(node_pair)
                                pairs.append(node_pair)

                if not pairs:
                    return None, counts_by_level

                next_matches[eid] = tuple(pairs)
                self._update_allowed_nodes(allowed, edge.source, (p[0] for p in pairs))
                if allowed[edge.source] == set():
                    return None, counts_by_level
                self._update_allowed_nodes(allowed, edge.target, (p[1] for p in pairs))
                if allowed[edge.target] == set():
                    return None, counts_by_level

            current = next_matches
            counts_by_level.append({eid: len(current[eid]) for eid in edge_ids})

        return current, counts_by_level

    @staticmethod
    def _update_allowed_nodes(
        allowed: MutableMapping[VertexId, Optional[Set[OctreeNode]]],
        vertex: VertexId,
        nodes: Iterable[OctreeNode],
    ) -> None:
        node_set = set(nodes)
        if allowed[vertex] is None:
            allowed[vertex] = node_set
        else:
            allowed[vertex] &= node_set

    def _edge_order(
        self, pattern: SpatialPattern, edge_ids: Sequence[int], previous_counts: Mapping[int, int]
    ) -> List[int]:
        """Heuristic order: exclusive edges first, then fewer prior matches."""

        return sorted(
            edge_ids,
            key=lambda eid: (
                0 if pattern.edges[eid].sign.is_exclusive else 1,
                previous_counts.get(eid, inf),
                eid,
            ),
        )

    def _node_pair_is_nmatch(
        self, pattern: SpatialPattern, eid: int, src: OctreeNode, tgt: OctreeNode, *, level: int
    ) -> bool:
        edge = pattern.edges[eid]
        lower2 = edge.lower * edge.lower
        upper2 = edge.upper * edge.upper
        if min_distance_sq_boxes(src.mbr, tgt.mbr) > upper2 + EPS:
            return False
        if max_distance_sq_boxes(src.mbr, tgt.mbr) < lower2 - EPS:
            return False

        sign = edge.sign
        if sign in (EdgeSign.SOURCE_EXCLUDES_TARGET, EdgeSign.MUTUAL_EXCLUSION):
            target_keyword = pattern.vertices[edge.target]
            if self._has_forbidden_node(src.mbr, target_keyword, level, edge.lower, except_node=tgt):
                return False
        if sign in (EdgeSign.TARGET_EXCLUDES_SOURCE, EdgeSign.MUTUAL_EXCLUSION):
            source_keyword = pattern.vertices[edge.source]
            if self._has_forbidden_node(tgt.mbr, source_keyword, level, edge.lower, except_node=src):
                return False
        return True

    def _has_forbidden_node(
        self, source_mbr: Box3D, target_keyword: str, level: int, lower: float, *, except_node: OctreeNode
    ) -> bool:
        """Conservative node-level test for exclusion lower bounds."""

        if lower <= 0:
            return False
        tree = self.index.tree_for_keyword(target_keyword)
        assert tree is not None
        threshold = lower * lower
        for node in tree.nodes_at_level(level):
            if node is except_node:
                continue
            if max_distance_sq_boxes(source_mbr, node.mbr) < threshold - EPS:
                return True
        return False

    # ---------- Object-level edge matching ----------

    def _identify_skip_edges(self, pattern: SpatialPattern, order: Sequence[int]) -> Set[int]:
        dsu = DisjointSet(pattern.vertices.keys())
        skip_edges: Set[int] = set()
        for eid in order:
            edge = pattern.edges[eid]
            if edge.sign is EdgeSign.INCLUSION and dsu.connected(edge.source, edge.target):
                skip_edges.add(eid)
            else:
                dsu.union(edge.source, edge.target)
        return skip_edges

    def _compute_ematches_for_edge(
        self,
        pattern: SpatialPattern,
        eid: int,
        node_pairs: Sequence[Tuple[OctreeNode, OctreeNode]],
    ) -> EdgeMatchStore:
        edge = pattern.edges[eid]
        pairs: List[Tuple[ObjectId, ObjectId]] = []
        seen: Set[Tuple[ObjectId, ObjectId]] = set()
        for src_node, tgt_node in node_pairs:
            for src_obj in src_node.objects:
                for tgt_obj in tgt_node.objects:
                    if self.require_distinct_objects and src_obj.id == tgt_obj.id:
                        continue
                    pair = (src_obj.id, tgt_obj.id)
                    if pair in seen:
                        continue
                    if self._object_pair_satisfies(pattern, edge, src_obj, tgt_obj):
                        seen.add(pair)
                        pairs.append(pair)
        return EdgeMatchStore.from_pairs(pairs)

    def _object_pair_satisfies(
        self, pattern: SpatialPattern, edge: PatternEdge, src_obj: SpatialObject, tgt_obj: SpatialObject
    ) -> bool:
        d2 = squared_distance_points(src_obj.point, tgt_obj.point)
        lower2 = edge.lower * edge.lower
        upper2 = edge.upper * edge.upper
        if d2 < lower2 - EPS or d2 > upper2 + EPS:
            return False

        if edge.sign in (EdgeSign.SOURCE_EXCLUDES_TARGET, EdgeSign.MUTUAL_EXCLUSION):
            target_keyword = pattern.vertices[edge.target]
            tree = self.index.tree_for_keyword(target_keyword)
            assert tree is not None
            if tree.any_point_within_radius(src_obj.point, edge.lower):
                return False

        if edge.sign in (EdgeSign.TARGET_EXCLUDES_SOURCE, EdgeSign.MUTUAL_EXCLUSION):
            source_keyword = pattern.vertices[edge.source]
            tree = self.index.tree_for_keyword(source_keyword)
            assert tree is not None
            if tree.any_point_within_radius(tgt_obj.point, edge.lower):
                return False

        return True

    # ---------- Joining ----------

    def _join_order(
        self,
        edge_ids: Sequence[int],
        ematches: Mapping[int, EdgeMatchStore],
        n_counts: Mapping[int, int],
        skip_edges: Set[int],
    ) -> List[int]:
        non_skip = [eid for eid in edge_ids if eid not in skip_edges]
        skip = [eid for eid in edge_ids if eid in skip_edges]
        return sorted(non_skip, key=lambda eid: (len(ematches[eid]), eid)) + sorted(
            skip, key=lambda eid: (n_counts[eid], eid)
        )

    def _extend_with_edge(
        self,
        assignments: Sequence[Dict[VertexId, ObjectId]],
        edge: PatternEdge,
        store: EdgeMatchStore,
        limit: Optional[int],
    ) -> List[Dict[VertexId, ObjectId]]:
        out: List[Dict[VertexId, ObjectId]] = []
        for assignment in assignments:
            for src_id, tgt_id in self._candidate_pairs(assignment, edge, store):
                extended = self._try_assign_pair(assignment, edge.source, src_id, edge.target, tgt_id)
                if extended is not None:
                    out.append(extended)
                    if limit is not None and len(out) >= limit:
                        return out
        return out

    @staticmethod
    def _candidate_pairs(
        assignment: Mapping[VertexId, ObjectId], edge: PatternEdge, store: EdgeMatchStore
    ) -> Iterator[Tuple[ObjectId, ObjectId]]:
        has_src = edge.source in assignment
        has_tgt = edge.target in assignment
        if has_src and has_tgt:
            pair = (assignment[edge.source], assignment[edge.target])
            if pair in store.pair_set:
                yield pair
            return
        if has_src:
            src_id = assignment[edge.source]
            for tgt_id in store.by_source.get(src_id, ()):
                yield (src_id, tgt_id)
            return
        if has_tgt:
            tgt_id = assignment[edge.target]
            for src_id in store.by_target.get(tgt_id, ()):
                yield (src_id, tgt_id)
            return
        yield from store.pairs

    def _try_assign_pair(
        self,
        assignment: Mapping[VertexId, ObjectId],
        src_vertex: VertexId,
        src_id: ObjectId,
        tgt_vertex: VertexId,
        tgt_id: ObjectId,
    ) -> Optional[Dict[VertexId, ObjectId]]:
        extended = dict(assignment)
        if not self._put_assignment(extended, src_vertex, src_id):
            return None
        if not self._put_assignment(extended, tgt_vertex, tgt_id):
            return None
        return extended

    def _put_assignment(self, assignment: Dict[VertexId, ObjectId], vertex: VertexId, object_id: ObjectId) -> bool:
        if vertex in assignment:
            return assignment[vertex] == object_id
        if self.require_distinct_objects and object_id in assignment.values():
            return False
        assignment[vertex] = object_id
        return True

    def _filter_by_skip_edge(
        self, assignments: Sequence[Dict[VertexId, ObjectId]], edge: PatternEdge, limit: Optional[int]
    ) -> List[Dict[VertexId, ObjectId]]:
        out: List[Dict[VertexId, ObjectId]] = []
        lower2 = edge.lower * edge.lower
        upper2 = edge.upper * edge.upper
        for assignment in assignments:
            if edge.source not in assignment or edge.target not in assignment:
                # This should not occur for a skip edge after all non-skip edges,
                # but keeping it explicit makes the behavior easy to inspect.
                continue
            src_obj = self.index.object(assignment[edge.source])
            tgt_obj = self.index.object(assignment[edge.target])
            d2 = squared_distance_points(src_obj.point, tgt_obj.point)
            if lower2 - EPS <= d2 <= upper2 + EPS:
                out.append(dict(assignment))
                if limit is not None and len(out) >= limit:
                    return out
        return out

    def _extend_isolated_vertices(
        self,
        assignments: Sequence[Dict[VertexId, ObjectId]],
        pattern: SpatialPattern,
        limit: Optional[int],
    ) -> List[Dict[VertexId, ObjectId]]:
        out = [dict(a) for a in assignments]
        for vertex, keyword in pattern.vertices.items():
            if all(vertex in a for a in out):
                continue
            candidates = self.index.objects_for_keyword(keyword)
            next_out: List[Dict[VertexId, ObjectId]] = []
            for assignment in out:
                if vertex in assignment:
                    next_out.append(assignment)
                    continue
                for obj in candidates:
                    if self.require_distinct_objects and obj.id in assignment.values():
                        continue
                    extended = dict(assignment)
                    extended[vertex] = obj.id
                    next_out.append(extended)
                    if limit is not None and len(next_out) >= limit:
                        break
                if limit is not None and len(next_out) >= limit:
                    break
            out = next_out
            if not out:
                return []
        return out[:limit] if limit is not None else out


def brute_force_match(
    objects: Iterable[SpatialObject],
    pattern: SpatialPattern,
    *,
    require_distinct_objects: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[VertexId, ObjectId]]:
    """Small-data reference implementation used for testing.

    It checks the same semantics as ESPM3DMatcher without using an index.
    """

    object_by_id = {obj.id: obj for obj in objects}
    by_keyword: DefaultDict[str, List[SpatialObject]] = defaultdict(list)
    for obj in object_by_id.values():
        for keyword in obj.keywords:
            by_keyword[keyword].append(obj)

    vertices = list(pattern.vertices.keys())
    out: List[Dict[VertexId, ObjectId]] = []

    def any_within(keyword: str, point: Point3D, radius: float) -> bool:
        if radius <= 0:
            return False
        threshold = radius * radius
        return any(squared_distance_points(point, obj.point) < threshold - EPS for obj in by_keyword.get(keyword, ()))

    def edge_ok(edge: PatternEdge, assignment: Mapping[VertexId, ObjectId]) -> bool:
        src_obj = object_by_id[assignment[edge.source]]
        tgt_obj = object_by_id[assignment[edge.target]]
        d2 = squared_distance_points(src_obj.point, tgt_obj.point)
        if d2 < edge.lower * edge.lower - EPS or d2 > edge.upper * edge.upper + EPS:
            return False
        if edge.sign in (EdgeSign.SOURCE_EXCLUDES_TARGET, EdgeSign.MUTUAL_EXCLUSION):
            if any_within(pattern.vertices[edge.target], src_obj.point, edge.lower):
                return False
        if edge.sign in (EdgeSign.TARGET_EXCLUDES_SOURCE, EdgeSign.MUTUAL_EXCLUSION):
            if any_within(pattern.vertices[edge.source], tgt_obj.point, edge.lower):
                return False
        return True

    def recurse(i: int, assignment: Dict[VertexId, ObjectId]) -> None:
        if limit is not None and len(out) >= limit:
            return
        if i == len(vertices):
            if all(edge_ok(edge, assignment) for edge in pattern.edges):
                out.append(dict(assignment))
            return
        vertex = vertices[i]
        keyword = pattern.vertices[vertex]
        for obj in by_keyword.get(keyword, ()):
            if require_distinct_objects and obj.id in assignment.values():
                continue
            assignment[vertex] = obj.id
            # Early check fully assigned incident edges.
            ok = True
            for edge in pattern.edges:
                if edge.source in assignment and edge.target in assignment and not edge_ok(edge, assignment):
                    ok = False
                    break
            if ok:
                recurse(i + 1, assignment)
            del assignment[vertex]

    recurse(0, {})
    return out
