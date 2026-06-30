# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides methods for converting ZX-graphs into numpy tensors
and using these tensors to test semantic equality of ZX-graphs.
This module is not meant as an efficient quantum simulator.
Due to the way the tensor is calculated it can only handle
circuits of small size before running out of memory on a regular machine.
Currently, it can reliably transform 9 qubit circuits into tensors.
If the ZX-diagram is not circuit-like, but instead has nodes with high degree,
it will run out of memory even sooner."""

__all__ = ['tensorfy', 'compare_tensors', 'compose_tensors',
            'adjoint', 'is_unitary','tensor_to_matrix',
            'find_scalar_correction']

import itertools
from math import pi, sqrt

from typing import Optional

from .symbolic import Poly


import numpy as np
np.set_printoptions(suppress=True)

# typing imports
from typing import TYPE_CHECKING, List, Dict, Union, Tuple
from numpy.typing import NDArray
from .utils import FractionLike, FloatInt, VertexType, EdgeType, get_z_box_label, settings
if TYPE_CHECKING:
    from .graph.base import BaseGraph, VT, ET
    from .circuit import Circuit
TensorConvertible = Union[np.ndarray, 'Circuit', 'BaseGraph']

def Z_box_to_tensor(arity: int, parameter: complex) -> np.ndarray:
    m = np.zeros([2]*arity, dtype = complex)
    if arity == 0:
        m[()] = 1 + parameter
        return m
    m[(0,)*arity] = 1
    m[(1,)*arity] = parameter
    return m

def Z_to_tensor(arity: int, phase: float) -> np.ndarray:
    return Z_box_to_tensor(arity, np.exp(1j*phase))

def X_to_tensor(arity: int, phase: float) -> np.ndarray:
    if arity == 0:
      m = np.zeros([2]*arity, dtype = complex)
      m[()] = 1 + np.exp(1j*phase)
      return m
    m = np.ones(2**arity, dtype = complex)
    for i in range(2**arity):
        if bin(i).count("1")%2 == 0:
            m[i] += np.exp(1j*phase)
        else:
            m[i] -= np.exp(1j*phase)
    return np.power(np.sqrt(0.5),arity)*m.reshape([2]*arity)

def H_to_tensor(arity: int, phase: float, label: Optional[complex] = None) -> np.ndarray:
    m = np.ones(2**arity, dtype = complex)
    if label is not None:
        m[-1] = label
    elif phase != 0:
        m[-1] = np.exp(1j*phase)
    return m.reshape([2]*arity)

def W_to_tensor(arity: int) -> np.ndarray:
    m = np.zeros([2]*arity,dtype=complex)
    if arity == 0:
        return m
    for i in range(arity):
        index = [0,]*arity
        index[i] = 1
        m[tuple(index)] = 1
    return m

def pop_and_shift(verts, indices):
    res = [indices[v].pop() for v in verts if v in indices]
    for i in sorted(res,reverse=True):
        for w,l in indices.items():
            l2 = []
            for j in l:
                if j>i: l2.append(j-1)
                else: l2.append(j)
            indices[w] = l2
    return res

# Vertex types only the naive backend can tensorfy: rank-width's full_reduce/leaf code
# rejects them, so 'auto' routes any diagram containing them to naive.
_RW_BLOCKERS = {VertexType.H_BOX, VertexType.W_INPUT, VertexType.W_OUTPUT, VertexType.Z_BOX}

# log2(flops) below which raw-naive is cheap enough that 'auto' skips the (otherwise
# dominated) full_reduce probe and runs it directly. Optimization gate only: a wrong value
# costs at most one wasted full_reduce, never correctness.
_FAST_ENOUGH = 22


def tensorfy(g: 'BaseGraph[VT,ET]',
             preserve_scalar: bool = True,
             strategy: str = 'naive',
             verbose: bool = False,
             max_memory: Optional[int] = None) -> NDArray[np.complex128]:
    """
    Returns a multidimensional numpy array representing the linear map the ZX diagram implements.
    Available simulation strategies are:

    - 'naive': good for sparse graphs
    - 'auto': estimate naive's peak memory (cheaply and exactly) on both the diagram and its
      ``full_reduce``d form, run naive on whichever is cheaper and fits ``max_memory``, and
      fall back to rank-width only when neither does. Raises ``MemoryError`` reporting the
      predicted sizes if no backend fits the budget.
    - 'rw-greedy-b2t': rank-width with greedy bottom-to-top heuristic
    - 'rw-greedy-linear': rank-width with greedy-linear heuristic
    - 'rw-auto': choose the best of 'rw-greedy-b2t' and 'rw-greedy-linear'

    Args:
        g: ZX diagram
        preserve_scalar: whether to account for the diagram scalar. With ``strategy='auto'``
            the result can differ from ``'naive'`` by a global scalar whenever the reduced
            graph or rank-width is used (the same convention as ``'rw-*'``); compare with
            :func:`compare_tensors` rather than ``numpy.allclose`` when ``preserve_scalar=False``.
        strategy: which simulation strategy to use
        verbose: print additional info (``'auto'`` prints the path it chose)
        max_memory: only used by ``strategy='auto'``: byte budget for the largest intermediate
            tensor. ``None`` (default) detects a conservative fraction of available RAM (via
            ``psutil`` if installed, else a static fallback). ``'auto'`` raises ``ValueError``
            on diagrams with symbolic (``Poly``) phases, which no backend can tensorfy.

    Returns:
        Numpy tensor having (num_inputs + num_outputs) dimensions (output dimensions first)
    """
    if g.is_hybrid():
        raise ValueError("Hybrid graphs are not supported.")
    if strategy == 'naive':
        return tensorfy_naive(g, preserve_scalar=preserve_scalar)
    elif strategy == 'auto':
        return _tensorfy_auto(g, preserve_scalar=preserve_scalar,
                              max_memory=max_memory, verbose=verbose)
    elif strategy.startswith('rw-'):
        from .rank_width import tensorfy_rw
        return tensorfy_rw(g, strategy=strategy, preserve_scalar=preserve_scalar, verbose=verbose)
    else:
        raise ValueError('Unknown simulation strategy')

def tensorfy_naive(g: 'BaseGraph[VT,ET]', preserve_scalar: bool = True) -> NDArray[np.complex128]:
    rows = g.rows()
    phases = g.phases()
    types = g.types()
    depth = g.depth()
    verts_row: Dict[FloatInt, List['VT']] = {}
    for v in g.vertices():
        r = rows[v]
        if r in verts_row: verts_row[r].append(v)
        else: verts_row[r] = [v]

    inputs = g.inputs()
    outputs = g.outputs()
    if not inputs and not outputs:
        if any(g.type(v)==VertexType.BOUNDARY for v in g.vertices()):
            raise ValueError("Diagram contains BOUNDARY-type vertices, but has no inputs or outputs set. Perhaps call g.auto_detect_io() first?")

    had = 1/sqrt(2)*np.array([[1,1],[1,-1]])
    id2 = np.identity(2)
    tensor = np.array(1.0,dtype='complex128')
    qubits = len(inputs)
    for i in range(qubits): tensor = np.tensordot(tensor,id2,axes=0)
    inputs = tuple(inputs)
    indices = {}
    for i, v in enumerate(inputs):
        indices[v] = [1 + 2*i]

    for i,r in enumerate(sorted(verts_row.keys())):
        for v in sorted(verts_row[r]):
            if types[v] == VertexType.DUMMY:
                continue
            incident = list(g.incident_edges(v))
            neigh = list(itertools.chain.from_iterable(
                set(g.edge_st(e)) - {v} for e in incident
            ))
            self_loops = [e for e in incident if g.edge_s(e) == g.edge_t(e)]
            d = len(neigh) + len(self_loops) * 2
            if v in inputs:
                if types[v] != VertexType.BOUNDARY: raise ValueError("Wrong type for input:", v, types[v])
                continue # inputs already taken care of
            if v in outputs:
                if d != 1: raise ValueError("Weird output")
                if types[v] != VertexType.BOUNDARY: raise ValueError("Wrong type for output:",v, types[v])
                d += 1
                t = id2
            else:
                p = phases[v]
                if isinstance(p, Poly):
                    raise ValueError(f"Can't convert diagram with parameters to tensor: {str(p)}")
                phase = pi*p
                if types[v] == VertexType.Z:
                    t = Z_to_tensor(d,phase)
                elif types[v] == VertexType.X:
                    t = X_to_tensor(d,phase)
                elif types[v] == VertexType.H_BOX:
                    # Check if H-box has a complex label.
                    h_label = g.vdata(v, 'label', None)
                    if h_label is not None:
                        t = H_to_tensor(d, 0, label=complex(h_label))
                    else:
                        t = H_to_tensor(d, phase)
                elif types[v] == VertexType.W_INPUT or types[v] == VertexType.W_OUTPUT:
                    if phase != 0: raise ValueError("Phase on W node")
                    t = W_to_tensor(d)
                elif types[v] == VertexType.Z_BOX:
                    if phase != 0: raise ValueError("Phase on Z box")
                    label = get_z_box_label(g, v)
                    t = Z_box_to_tensor(d, label)
                else:
                    raise ValueError("Vertex %s has non-ZXH type but is not an input or output" % str(v))
            for sl in self_loops:
                if g.edge_type(sl) == EdgeType.HADAMARD:
                    t = np.tensordot(t,had)
                elif g.edge_type(sl) == EdgeType.SIMPLE:
                    t = np.trace(t)
                else:
                    raise NotImplementedError(f"Tensor contraction with {repr(sl)} self-loops is not implemented.")
            # Iterate over incident edges rather than neighbours so parallel
            # edges of different types are kept as distinct legs.
            leg_ety = []
            for e in incident:
                if g.edge_s(e) == g.edge_t(e): continue  # self-loop, handled above
                s, tgt = g.edge_st(e)
                n = tgt if s == v else s
                if rows[n] < r or (rows[n] == r and n < v):
                    leg_ety.append((n, g.edge_type(e)))
            leg_ety.sort(key=lambda ne: ne[1] == EdgeType.HADAMARD)
            nn = [n for n, _ in leg_ety]
            for _, et in leg_ety:
                if et == EdgeType.HADAMARD:
                    t = np.tensordot(t,had,(0,0)) # Hadamard edges are moved to the last index of t
            contr = pop_and_shift(nn,indices) #the last indices in contr correspond to hadamard contractions
            tensor = np.tensordot(tensor,t,axes=(contr,list(range(len(t.shape)-len(contr),len(t.shape)))))
            indices[v] = list(range(len(tensor.shape)-d+len(contr), len(tensor.shape)))

            if not preserve_scalar and i % 10 == 0:
                if np.abs(tensor).max() < 10**-6: # Values are becoming too small
                    tensor *= 10**4 # So scale all the numbers up
    perm = []
    for o in outputs:
        perm.append(indices[o][0])
    for i in range(len(inputs)):
        perm.append(i)

    tensor = np.transpose(tensor,perm)
    if preserve_scalar: tensor *= g.scalar.to_number()
    return tensor

def naive_cost_estimate(g: 'BaseGraph[VT,ET]') -> Tuple[int, float]:
    """Predict ``tensorfy_naive(g)``'s cost in O(V+E) without building any tensors.

    Returns ``(peak_rank, log2_flops)``, where ``peak_rank`` is the log2 of the largest
    intermediate the backend ever holds (so its memory is ``16 * 2**peak_rank`` bytes) and
    ``log2_flops`` is a log2 estimate of the total tensordot work. The contraction's tensor
    *shapes* depend only on the diagram's topology -- not on phases or vertex colour -- so this
    dry-run of ``tensorfy_naive``'s row-order contraction is exact for the dense-tensordot
    backend.

    ``peak_rank`` is the max over each step of (a) the running accumulator and (b) the
    spider's own dense ``2**d`` vertex tensor; a high-degree spider can make the latter the
    larger one, so the accumulator alone would under-count (and so route an out-of-memory
    contraction to naive).
    """
    rows = g.rows()
    types = g.types()
    inputs = set(g.inputs())
    outputs = set(g.outputs())

    verts_row: Dict[FloatInt, List['VT']] = {}
    for v in g.vertices():
        verts_row.setdefault(rows[v], []).append(v)

    frontier = 2 * len(inputs)        # accumulator seeded with one id2 per input
    peak = frontier
    log_flops = float("-inf")         # log2(0)

    for r in sorted(verts_row):
        for v in sorted(verts_row[r]):
            if types[v] == VertexType.DUMMY:
                continue
            if v in inputs:
                continue
            incident = list(g.incident_edges(v))
            non_self = [e for e in incident if g.edge_s(e) != g.edge_t(e)]
            m = len(non_self)
            s = len(incident) - m                          # self-loop edges
            vt_arity = 0 if v in outputs else m + 2 * s     # dense vertex tensor (id2 for outputs)
            if v in outputs:
                m += 1                                      # output pads one external leg
            c = 0
            for e in non_self:
                a, b = g.edge_st(e)
                n = b if a == v else a
                if rows[n] < r or (rows[n] == r and n < v):
                    c += 1
            log_flops = np.logaddexp2(log_flops, frontier + m - c)
            frontier = frontier + m - 2 * c
            peak = max(peak, frontier, vt_arity)            # accumulator AND vertex tensor

    return peak, float(log_flops)

def _default_max_memory() -> int:
    """Byte budget for ``strategy='auto'`` when ``max_memory`` is not given:
    ``pyzx.settings.tensor_auto_memory_fraction`` of available RAM (via the optional
    :mod:`psutil` dependency), or ``pyzx.settings.tensor_auto_max_memory_fallback`` bytes
    when psutil is not installed."""
    try:
        import psutil  # type: ignore[import-untyped]
        return int(psutil.virtual_memory().available * settings.tensor_auto_memory_fraction)
    except ImportError:
        return settings.tensor_auto_max_memory_fallback

def _tensorfy_auto(g: 'BaseGraph[VT,ET]',
                   preserve_scalar: bool = True,
                   max_memory: Optional[int] = None,
                   verbose: bool = False) -> NDArray[np.complex128]:
    """Choose a backend for ``g`` by predicted peak memory and run it (see :func:`tensorfy`
    with ``strategy='auto'``). The whole decision is in one currency -- peak memory in bytes --
    with no fitted speed constants. ``max_memory=None`` uses :func:`_default_max_memory`."""
    from .simplify import full_reduce
    from .rank_width import (greedy_b2t_decomposition, greedy_linear_order,
                             linear_decomposition, rank_score_flops, rw_peak_exact,
                             tensorfy_rw)

    budget = _default_max_memory() if max_memory is None else max_memory

    def fits(peak: int) -> bool:
        return 16 * 2 ** peak <= budget

    # Tier 1 -- capability gate. No backend handles symbolic phases; only naive handles
    # H_BOX/W/Z_BOX, so a diagram containing them is forced to naive (still memory-guarded).
    phases = g.phases()
    types = g.types()
    blocked = False
    for v in g.vertices():
        if isinstance(phases[v], Poly):
            raise ValueError(f"Can't convert diagram with parameters to tensor: {phases[v]}")
        if types[v] in _RW_BLOCKERS:
            blocked = True
    if blocked:
        peak, _ = naive_cost_estimate(g)
        if not fits(peak):
            raise MemoryError(f"diagram needs ~16*2^{peak} bytes; H_BOX/W/Z_BOX force the "
                              f"naive backend (rank-width cannot handle them)")
        if verbose:
            print(f"auto: rw-blocked vertex type present -> naive (peak={peak})")
        return tensorfy_naive(g, preserve_scalar=preserve_scalar)

    # Tier 2 -- exact frontier estimate on the raw graph.
    peak_raw, flops_raw = naive_cost_estimate(g)
    if fits(peak_raw) and flops_raw <= _FAST_ENOUGH:
        if verbose:
            print(f"auto: raw-naive fits & cheap (peak={peak_raw}, flops={flops_raw:.1f}) -> naive")
        return tensorfy_naive(g, preserve_scalar=preserve_scalar)

    # Raw-naive is big or slow: does full_reduce give a cheaper naive? full_reduce crushes
    # tangled/closed diagrams but makes circuits worse, so estimate both and pick the lighter.
    g_red = g.copy()
    full_reduce(g_red)
    peak_red, _ = naive_cost_estimate(g_red)

    naive_opts = [(peak_raw, g), (peak_red, g_red)]
    feasible = [(p, gg) for (p, gg) in naive_opts if fits(p)]
    if feasible:
        p, gg = min(feasible, key=lambda o: o[0])
        if verbose:
            print(f"auto: naive on {'reduced' if gg is g_red else 'raw'} (peak={p}) -> naive")
        return tensorfy_naive(gg, preserve_scalar=preserve_scalar)

    # Neither naive variant fits -> rank-width (guarded; rw can OOM too). Build both heuristic
    # decompositions, take each one's exact peak, and choose budget-aware -- rw-auto's flops
    # choice is sometimes the memory-heavier decomposition.
    decomps = [greedy_b2t_decomposition(g_red),
               linear_decomposition(greedy_linear_order(g_red))]
    cands = [(rw_peak_exact(d, g_red), rank_score_flops(d, g_red), d) for d in decomps]
    fitting = [c for c in cands if fits(c[0])]
    rw_peak, _, rw_decomp = (min(fitting, key=lambda c: c[1]) if fitting    # fits -> fastest plan
                             else min(cands, key=lambda c: c[0]))           # none fit -> lightest

    options = [(peak_raw, 'naive', g), (peak_red, 'naive', g_red), (rw_peak, 'rw', g_red)]
    peak_min, backend, graph_min = min(options, key=lambda o: o[0])   # ties -> naive (list order)
    if fits(peak_min):
        if verbose:
            print(f"auto: both naive over budget; lightest = {backend} (peak={peak_min}) -> {backend}")
        if backend == 'rw':
            return tensorfy_rw(g_red, decomp=rw_decomp, skip_reduce=True,
                               preserve_scalar=preserve_scalar, verbose=verbose)
        return tensorfy_naive(graph_min, preserve_scalar=preserve_scalar)
    raise MemoryError(f"diagram needs ~16*2^{peak_min} bytes in the lightest backend "
                      f"(naive raw 2^{peak_raw} / reduced 2^{peak_red} / rw 2^{rw_peak}); "
                      f"raise max_memory or simplify the diagram first")

def tensor_to_matrix(t: np.ndarray, inputs: int, outputs: int) -> np.ndarray:
    """Takes a tensor generated by ``tensorfy`` and turns it into a matrix.
    The ``inputs`` and ``outputs`` arguments specify the final shape of the matrix:
    2^(outputs) x 2^(inputs)"""
    rows = []
    for r in range(2**outputs):
        if outputs == 0:
            o = []
        else:
            o = [int(i) for i in bin(r)[2:].zfill(outputs)]
        row = []
        if inputs == 0:
            row.append(t[tuple(o)])
        else:
            for c in range(2**inputs):
                a = o.copy()
                a.extend([int(i) for i in bin(c)[2:].zfill(inputs)])
                row.append(t[tuple(a)])
        rows.append(row)
    return np.array(rows)

def compare_tensors(t1: TensorConvertible,t2: TensorConvertible,
                    preserve_scalar: bool=False, strategy: str='naive') -> bool:
    """Returns true if ``t1`` and ``t2`` represent equal tensors by calling :func:`~pyzx.tensor.tensorfy`.
    When `preserve_scalar` is False (the default), equality is checked up to nonzero rescaling.

    Example: To check whether two ZX-graphs `g1` and `g2` are semantically the same you would do::

        compare_tensors(g1,g2) # True if g1 and g2 represent the same linear map up to nonzero scalar

    """
    from .circuit import Circuit

    if not isinstance(t1, np.ndarray):
        t1 = t1.to_tensor(preserve_scalar, strategy)
    if not isinstance(t2, np.ndarray):
        t2 = t2.to_tensor(preserve_scalar, strategy)
    if np.allclose(t1,t2): return True
    if preserve_scalar: return False # We do not check for equality up to scalar
    epsilon = 10**-14
    for i,a in enumerate(t1.flat):
        if abs(a)>epsilon:
            if abs(t2.flat[i])<epsilon: return False
            break
    else:
        raise ValueError("Tensor is too close to zero")
    return np.allclose(t1/a,t2/t2.flat[i])

def find_scalar_correction(t1: TensorConvertible, t2:TensorConvertible) -> complex:
    """Returns the complex number ``z`` such that ``t1 = z*t2``.

    Warning:
        This function assumes that ``compare_tensors(t1,t2,preserve_scalar=False)`` is True,
        i.e. that ``t1`` and ``t2`` indeed are equal up to global scalar.
        If they aren't, this function returns garbage.

    """
    if not isinstance(t1, np.ndarray):
        t1 = t1.to_tensor(preserve_scalar=True)
    if not isinstance(t2, np.ndarray):
        t2 = t2.to_tensor(preserve_scalar=True)

    epsilon = 10**-14
    for i,a in enumerate(t1.flat):
        if abs(a)>epsilon:
            if abs(t2.flat[i])<epsilon: return 0
            return a/t2.flat[i]

    return 0


def compose_tensors(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Returns a tensor that is the result of composing the tensors together as if they
    were representing circuits::

        t1 = tensorfy(circ1)
        t2 = tensorfy(circ2)
        circ1.compose(circ2)
        t3 = tensorfy(circ1)
        t4 = compose_tensors(t1,t2)
        compare_tensors(t3,t4) # This is True

    """

    if len(t1.shape) != len(t2.shape):
        raise TypeError("Tensors represent circuits of different amount of qubits, "
                        "{!s} vs {!s}".format(len(t1.shape)//2,len(t2.shape)//2))
    q = len(t1.shape)//2
    contr2 = [q+i for i in range(q)]
    contr1 = [i for i in range(q)]
    t = np.tensordot(t1,t2,axes=(contr1,contr2))
    transp = []
    for i in range(q):
        transp.append(q+i)
    for i in range(q):
        transp.append(i)
    return np.transpose(t,transp)


def adjoint(t: np.ndarray) -> np.ndarray:
    """Returns the adjoint of the tensor as if it were representing
    a circuit::

        t = tensorfy(circ)
        tadj = tensorfy(circ.adjoint())
        compare_tensors(adjoint(t),tadj) # This is True

    """

    q = len(t.shape)//2
    transp = []
    for i in range(q):
        transp.append(q+i)
    for i in range(q):
        transp.append(i)
    return np.transpose(t.conjugate(),transp)


def is_unitary(g: 'BaseGraph') -> bool:
    """Returns whether the given ZX-graph is equal to a unitary (up to a number)."""
    from .generate import identity # Imported here to prevent circularity
    adj = g.adjoint()
    adj.compose(g)
    return compare_tensors(adj.to_tensor(), identity(len(g.inputs()),2).to_tensor(), False)
