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


import unittest
import random
import sys
from types import ModuleType
from typing import Optional

if __name__ == '__main__':
    sys.path.append('..')
    sys.path.append('.')
import math

from pyzx.graph import Graph
from pyzx.graph.multigraph import Multigraph
from pyzx.generate import cliffords
from pyzx.circuit import Circuit
from pyzx.utils import VertexType, EdgeType, set_h_box_label

np: Optional[ModuleType]
try:
    import numpy as np
    from pyzx.tensor import (tensorfy, tensorfy_naive, naive_cost_estimate, compare_tensors,
                             compose_tensors, adjoint, H_to_tensor)
    from pyzx.rank_width import rw_peak_exact, tensorfy_rw
    from pyzx.simplify import full_reduce
    from pyzx.generate import cliffordT
    from pyzx.symbolic import new_var
except ImportError:
    np = None

SEED = 1337


@unittest.skipUnless(np, "numpy needs to be installed for this to run")
class TestTensor(unittest.TestCase):

    def test_scalar_difference(self):
        array = np.array([[0,1],[1,0]])
        scalar = 0.75 + 3j
        self.assertFalse(compare_tensors(scalar*array,array,True))
    def test_scalar_difference_ignore(self):
        array = np.array([[0,1],[1,0]])
        scalar = 0.75 + 3j
        self.assertTrue(compare_tensors(scalar*array,array,False))

    def test_trivial_inequality(self):
        array = np.array([[1,0],[0,1]])
        array2= np.array([[0,1],[1,0]])
        self.assertFalse(compare_tensors(array, array2))

    def test_id_graph(self):
        g = Graph()
        i = g.add_vertex(0,0,0)
        o = g.add_vertex(0,0,1)
        g.set_inputs((i,))
        g.set_outputs((o,))
        g.add_edge((i,o))
        t = tensorfy(g)
        id_array = np.array([[1,0],[0,1]])
        self.assertTrue(np.allclose(t,id_array))
        self.assertTrue(compare_tensors(t,id_array))

    def test_equality_of_id_zx_graph_to_id(self):
        g = Graph()
        i = g.add_vertex(0,0,0)
        o = g.add_vertex(0,0,2)
        g.set_inputs((i,))
        g.set_outputs((o,))
        g2 = g.copy()
        g.add_edge((i,o))
        v = g2.add_vertex(1,0,1)
        g2.add_edges([(i,v),(v,o)])
        tensor1 = tensorfy(g)
        tensor2 = tensorfy(g2)
        self.assertTrue(compare_tensors(tensor1,tensor2))

    def test_inequality_id_and_swap(self):
        g = Graph()
        i1 = g.add_vertex(0,0,0)
        i2 = g.add_vertex(0,1,0)
        o1 = g.add_vertex(0,0,1)
        o2 = g.add_vertex(0,1,1)
        g.set_inputs((i1, i2))
        g.set_outputs((o1, o2))
        g2 = g.copy()
        g.add_edges([(i1,o2),(i2,o1)])
        g2.add_edges([(i1,o1),(i2,o2)])
        id_id = tensorfy(g2)
        swap = tensorfy(g)
        self.assertFalse(compare_tensors(id_id,swap))

    def test_three_cnots_is_swap(self):
        g = Graph()
        i1 = g.add_vertex(0,0,0)
        i2 = g.add_vertex(0,1,0)
        o1 = g.add_vertex(0,0,1)
        o2 = g.add_vertex(0,1,1)
        g.set_inputs((i1, i2))
        g.set_outputs((o1, o2))
        g.add_edges([(i1,o2),(i2,o1)])
        swap = tensorfy(g)
        c = Circuit(2)
        c.add_gate("CNOT",0,1)
        c.add_gate("CNOT",1,0)
        c.add_gate("CNOT",0,1)
        three_cnots = tensorfy(c.to_graph())
        self.assertTrue(compare_tensors(swap,three_cnots))

    def test_compose(self):
        random.seed(SEED)
        circ1 = cliffords(3,15)
        circ2 = cliffords(3,20)
        t1 = tensorfy(circ1)
        t2 = tensorfy(circ2)
        comp1 = compose_tensors(t1,t2)
        circ1.compose(circ2)
        comp2 = tensorfy(circ1)
        self.assertTrue(compare_tensors(comp1,comp2))

    def test_adjoint(self):
        random.seed(SEED)
        circ = cliffords(3, 16)
        t = tensorfy(circ)
        t_adj = adjoint(t)
        circ_adj = tensorfy(circ.adjoint())
        self.assertTrue(compare_tensors(t_adj,circ_adj))
    
    def test_multigraph_auto_simplify_parallel_edges(self):
        g = Multigraph()
        g.set_auto_simplify(True)
        i1 = g.add_vertex(1,0,0)
        i2 = g.add_vertex(2,1,0)
        g.add_edges([(i1, i2)] * 3)
        self.assertTrue(compare_tensors(g, np.array([np.sqrt(2)**(-1)]), preserve_scalar=True))

    def test_multiedge_scalar(self):
        g = Multigraph()
        g.set_auto_simplify(False)
        i1 = g.add_vertex(1,0,0)
        i2 = g.add_vertex(2,1,0)
        g.add_edges([(i1, i2)] * 3)
        self.assertTrue(compare_tensors(g, np.array([np.sqrt(2)**(-1)]), preserve_scalar=True))

    def test_self_loop_scalar(self):
        g = Multigraph()
        g.set_auto_simplify(False)
        i1 = g.add_vertex(1,0,0)
        g.add_edge((i1, i1))
        self.assertTrue(compare_tensors(g, np.array([2]), preserve_scalar=True))
        g.add_edge((i1, i1), 2)
        self.assertTrue(compare_tensors(g, np.array([0]), preserve_scalar=True))

    def test_self_loop_state(self):
        g = Multigraph()
        g.set_auto_simplify(False)
        i0 = g.add_vertex(0,0,0)
        i1 = g.add_vertex(2,0,1)
        g.set_inputs((i0,))
        g.add_edge((i0, i1))
        self.assertTrue(compare_tensors(g, np.array([1,0])))
        g.add_edge((i1, i1), 2)
        self.assertTrue(compare_tensors(g, np.array([0,1])))

    def test_self_loop_and_parallel_edge_map(self):
        g = Multigraph()
        g.set_auto_simplify(False)
        i0 = g.add_vertex(0,0,0)
        i1 = g.add_vertex(2,0,1)
        i2 = g.add_vertex(1,0,2)
        i3 = g.add_vertex(0,0,3)
        g.set_inputs((i0,))
        g.set_outputs((i3,))
        g.add_edges([(i0, i1), (i1, i1)] + [(i1, i2)] * 2)
        g.add_edges([(i2, i2), (i2, i3)], 2)
        self.assertTrue(compare_tensors(g, np.array([[0,0],[1,0]])))

    def test_parallel_mixed_edges_map(self):
        """An X spider connected to a Z spider by one simple AND one Hadamard
        edge in parallel implements the X gate up to scalar; tensorfy must not
        treat the parallel pair as if both edges were the same type."""
        from pyzx.utils import EdgeType
        g = Multigraph()
        g.set_auto_simplify(False)
        b_in  = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        x     = g.add_vertex(VertexType.X,        qubit=0, row=1)
        b_out = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
        z     = g.add_vertex(VertexType.Z,        qubit=-1, row=1)
        g.add_edge((b_in, x),  EdgeType.SIMPLE)
        g.add_edge((x, b_out), EdgeType.SIMPLE)
        g.add_edge((x, z),     EdgeType.SIMPLE)
        g.add_edge((x, z),     EdgeType.HADAMARD)
        g.set_inputs((b_in,)); g.set_outputs((b_out,))
        self.assertTrue(compare_tensors(g, np.array([[0,1],[1,0]]),
                                        preserve_scalar=False))

    def test_to_tensor_equivalent(self):
        g = Graph()
        g.add_vertex(VertexType.Z, phase=1)
        g1 = Graph()
        g1.add_vertex(VertexType.X, phase=1)
        self.assertTrue(g.to_tensor() == g1.to_tensor())

    def test_h_to_tensor_with_label(self):
        """Test H_to_tensor with explicit complex label."""
        t = H_to_tensor(2, 0, label=3+4j)
        expected = np.array([[1, 1], [1, 3+4j]])
        self.assertTrue(np.allclose(t, expected))

    def test_h_to_tensor_standard_hadamard(self):
        """Test H_to_tensor for standard Hadamard (label=-1 or phase=pi)."""
        t_label = H_to_tensor(2, 0, label=-1)
        t_phase = H_to_tensor(2, math.pi)
        expected = np.array([[1, 1], [1, -1]])
        self.assertTrue(np.allclose(t_label, expected))
        self.assertTrue(np.allclose(t_phase, expected))

    def test_tensorfy_hbox_with_complex_label(self):
        """Test tensorfy with H-box having complex label."""
        g = Graph()
        i = g.add_vertex(VertexType.BOUNDARY, 0, 0)
        h = g.add_vertex(VertexType.H_BOX, 0, 1)
        o = g.add_vertex(VertexType.BOUNDARY, 0, 2)
        g.set_inputs((i,))
        g.set_outputs((o,))
        g.add_edge((i, h))
        g.add_edge((h, o))
        set_h_box_label(g, h, 1j)

        t = tensorfy(g)
        expected = np.array([[1, 1], [1, 1j]])
        self.assertTrue(np.allclose(t, expected))

    def test_tensorfy_hbox_with_standard_label(self):
        """Test tensorfy with H-box having standard label -1."""
        g = Graph()
        i = g.add_vertex(VertexType.BOUNDARY, 0, 0)
        h = g.add_vertex(VertexType.H_BOX, 0, 1)
        o = g.add_vertex(VertexType.BOUNDARY, 0, 2)
        g.set_inputs((i,))
        g.set_outputs((o,))
        g.add_edge((i, h))
        g.add_edge((h, o))
        set_h_box_label(g, h, -1)

        t = tensorfy(g)
        expected = np.array([[1, 1], [1, -1]])
        self.assertTrue(np.allclose(t, expected))

    def test_tensorfy_hbox_phase_and_label_equivalence(self):
        """Test that phase=1 and label=-1 produce same tensor."""
        g1 = Graph()
        i1 = g1.add_vertex(VertexType.BOUNDARY, 0, 0)
        h1 = g1.add_vertex(VertexType.H_BOX, 0, 1)
        o1 = g1.add_vertex(VertexType.BOUNDARY, 0, 2)
        g1.set_inputs((i1,))
        g1.set_outputs((o1,))
        g1.add_edge((i1, h1))
        g1.add_edge((h1, o1))
        g1.set_phase(h1, 1)

        g2 = Graph()
        i2 = g2.add_vertex(VertexType.BOUNDARY, 0, 0)
        h2 = g2.add_vertex(VertexType.H_BOX, 0, 1)
        o2 = g2.add_vertex(VertexType.BOUNDARY, 0, 2)
        g2.set_inputs((i2,))
        g2.set_outputs((o2,))
        g2.add_edge((i2, h2))
        g2.add_edge((h2, o2))
        set_h_box_label(g2, h2, -1)

        self.assertTrue(compare_tensors(g1, g2, preserve_scalar=True))


@unittest.skipUnless(np, "numpy needs to be installed for this to run")
class TestTensorAuto(unittest.TestCase):
    """Tests for tensorfy(strategy='auto') and its cost estimators."""

    GIB = 1 << 30

    # ---- fixtures (match the validated POC corpus) ----
    @staticmethod
    def _cnot():
        g = Graph()
        in0 = g.add_vertex(VertexType.BOUNDARY, 0, 0)
        in1 = g.add_vertex(VertexType.BOUNDARY, 1, 0)
        z = g.add_vertex(VertexType.Z, 0, 1)
        x = g.add_vertex(VertexType.X, 1, 1)
        o0 = g.add_vertex(VertexType.BOUNDARY, 0, 2)
        o1 = g.add_vertex(VertexType.BOUNDARY, 1, 2)
        for e in [(in0, z), (in1, x), (z, x), (z, o0), (x, o1)]:
            g.add_edge(e, EdgeType.SIMPLE)
        g.set_inputs((in0, in1)); g.set_outputs((o0, o1))
        return g

    @staticmethod
    def _dense_closed(n, m, seed):
        from fractions import Fraction
        random.seed(seed)
        g = Graph()
        for _ in range(n):
            g.add_vertex(VertexType.Z, phase=Fraction(random.randint(0, 7), 4))
        while g.num_edges() < m:
            u, v = random.sample(range(n), 2)
            if (u, v) not in g.edge_set() and (v, u) not in g.edge_set():
                g.add_edge((u, v), EdgeType.HADAMARD)
        return g

    @staticmethod
    def _clifford(q, d, p_t, seed):
        random.seed(seed); np.random.seed(seed)
        c = cliffordT(q, d, p_t=p_t)
        return c.to_graph() if hasattr(c, "to_graph") else c

    def _path(self, g, max_memory):
        """The verbose path 'auto' prints for graph g under the given budget."""
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tensorfy(g, strategy='auto', max_memory=max_memory, verbose=True)
        return buf.getvalue()

    @staticmethod
    def _measured_rw_peak(g):
        """log2 of the largest array a real tensorfy_rw run allocates for g."""
        import pyzx.rank_width as rwm
        biggest = [0]
        def note(a):
            if hasattr(a, "size"):
                biggest[0] = max(biggest[0], int(a.size))
        def wrap(f):
            def inner(*a, **k):
                out = f(*a, **k); note(out); return out
            return inner
        orig = (np.kron, np.fft.fftn, np.tensordot, rwm.apply_parity_map)
        np.kron, np.fft.fftn, np.tensordot = wrap(orig[0]), wrap(orig[1]), wrap(orig[2])
        rwm.apply_parity_map = wrap(orig[3])
        try:
            note(tensorfy_rw(g, strategy='rw-auto'))
        finally:
            np.kron, np.fft.fftn, np.tensordot = orig[0], orig[1], orig[2]
            rwm.apply_parity_map = orig[3]
        return round(math.log2(biggest[0])) if biggest[0] else 0

    # ---- estimators ----
    def test_naive_estimate_peak_is_exact(self):
        # peak is purely structural, so it matches the true peaks exactly
        self.assertEqual(naive_cost_estimate(self._cnot())[0], 5)
        self.assertEqual(naive_cost_estimate(self._clifford(6, 40, 0.2, 1337))[0], 13)
        self.assertEqual(naive_cost_estimate(self._clifford(8, 60, 0.2, 1337))[0], 17)
        self.assertEqual(naive_cost_estimate(self._dense_closed(14, 40, 1337))[0], 23)

    def test_rw_peak_exact_matches_real_run(self):
        from pyzx.rank_width import (greedy_b2t_decomposition, linear_decomposition,
                                     greedy_linear_order, rank_score_flops)
        g = self._dense_closed(14, 40, 1337)
        gr = g.copy(); full_reduce(gr)
        decomps = {'b2t': greedy_b2t_decomposition(gr),
                   'linear': linear_decomposition(greedy_linear_order(gr))}
        flops = {k: rank_score_flops(d, gr) for k, d in decomps.items()}
        pick = min(flops, key=lambda k: flops[k])      # rw-auto chooses min flops
        self.assertEqual(rw_peak_exact(decomps[pick], gr), self._measured_rw_peak(g))

    # ---- correctness ----
    def test_auto_matches_naive_circuit(self):
        g = self._clifford(8, 60, 0.2, 1337)
        self.assertTrue(compare_tensors(tensorfy(g, strategy='auto', max_memory=self.GIB),
                                        tensorfy_naive(g), preserve_scalar=False))

    def test_auto_matches_naive_dense(self):
        g = self._dense_closed(14, 40, 1337)
        self.assertTrue(compare_tensors(tensorfy(g, strategy='auto', max_memory=self.GIB),
                                        tensorfy_naive(g), preserve_scalar=False))

    def test_reduced_naive_equals_raw_naive(self):
        # the invariant that licenses the full_reduce candidate (up to global scalar)
        for g in (self._clifford(6, 40, 0.2, 1337), self._dense_closed(14, 40, 1337)):
            gr = g.copy(); full_reduce(gr)
            self.assertTrue(compare_tensors(tensorfy_naive(gr), tensorfy_naive(g),
                                            preserve_scalar=False))

    def test_auto_via_graph_wrapper(self):
        g = self._clifford(6, 40, 0.2, 1337)
        self.assertTrue(compare_tensors(g.to_tensor(strategy='auto', max_memory=self.GIB),
                                        tensorfy_naive(g), preserve_scalar=False))

    # ---- capability gate ----
    def test_gate_hbox_falls_back_to_naive(self):
        g = Graph()
        i = g.add_vertex(VertexType.BOUNDARY, 0, 0)
        h = g.add_vertex(VertexType.H_BOX, 0, 1)
        o = g.add_vertex(VertexType.BOUNDARY, 0, 2)
        g.set_inputs((i,)); g.set_outputs((o,))
        g.add_edge((i, h)); g.add_edge((h, o))
        t = tensorfy(g, strategy='auto', max_memory=self.GIB)   # must not raise
        self.assertTrue(compare_tensors(t, tensorfy_naive(g), preserve_scalar=False))

    def test_gate_symbolic_phase_raises(self):
        g = Graph()
        a = new_var('a', is_bool=False, registry=g.var_registry)
        i = g.add_vertex(VertexType.BOUNDARY, 0, 0)
        z = g.add_vertex(VertexType.Z, 0, 1, phase=a)
        o = g.add_vertex(VertexType.BOUNDARY, 0, 2)
        g.set_inputs((i,)); g.set_outputs((o,))
        g.add_edge((i, z)); g.add_edge((z, o))
        with self.assertRaises(ValueError):
            tensorfy(g, strategy='auto', max_memory=self.GIB)

    # ---- selector paths ----
    def test_path_circuit_uses_raw_fastpath(self):
        self.assertIn("raw-naive fits & cheap", self._path(self._cnot(), self.GIB))

    def test_path_dense_uses_reduced(self):
        self.assertIn("naive on reduced", self._path(self._dense_closed(14, 40, 1337), self.GIB))

    def test_path_raw_over_budget_rescued_to_reduced(self):
        # raw peak over budget, reduced peak fits -> naive(reduced), NOT escalated to rw
        self.assertIn("naive on reduced", self._path(self._dense_closed(22, 90, 2024), self.GIB))

    def test_path_rw_branch_when_naive_infeasible(self):
        # tiny budget: both naive variants over budget but rw fits -> rw branch runs
        g = self._dense_closed(14, 40, 1337)
        self.assertIn("-> rw", self._path(g, 1024))
        self.assertTrue(compare_tensors(tensorfy(g, strategy='auto', max_memory=1024),
                                        tensorfy_naive(g), preserve_scalar=False))

    def test_over_budget_raises_clear_memoryerror(self):
        with self.assertRaises(MemoryError) as cm:
            tensorfy(self._dense_closed(14, 40, 1337), strategy='auto', max_memory=16)
        self.assertIn("rw 2^", str(cm.exception))      # message names all three peaks

    def test_unknown_strategy_still_raises(self):
        with self.assertRaises(ValueError):
            tensorfy(self._cnot(), strategy='bogus')


if __name__ == '__main__':
    unittest.main()
