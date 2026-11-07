"""Bounded restoration fixtures; run with unittest (no pytest dependency)."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'implementation' / 'src'))
import numpy as np
import jax.numpy as jnp
from besp.algorithms.jax import deterministic as d
from besp.runners import deterministic_dispatch as dispatch


def control_step(solver, candidate, policy=0):
    """Inject only the surrogate planning result; retain real true-model checks."""
    real_einsum = jnp.einsum
    calls = 0
    def surrogate_einsum(spec, model, value):
        nonlocal calls
        if model is solver.p_hat:
            calls += 1
            if calls == 1:
                return solver._last_expected_v_true
            h = solver.h
            r_tilde = solver.v[:, None]/(1+h) + h*solver.r/(1+h)
            gamma_eff = h*solver.gamma/(1+h)
            scores = jnp.full_like(solver.r, candidate)
            if calls > 2:
                scores = scores.at[:, policy].add(1.)
            return (scores-r_tilde)/gamma_eff
        return real_einsum(spec, model, value)
    with patch.object(d.jnp, 'einsum', side_effect=surrogate_einsum):
        return solver.step_with_policy()


def control_solver(**kwargs):
    return d.WaterfallISP_Control([[[1.], [1.]]], [[[1.], [1.]]], [[1., .9]], .5,
                                 inner_vi_max_iters=1, h_init=8., h_min=1.,
                                 stall_patience=kwargs.pop('stall_patience', 2),
                                 decay_factor=.5, **kwargs)


def batch_kwargs(n=1, patience=2, h=8., floor=1., max_queries=4):
    return dict(h_init=jnp.full(n, h), h_min=jnp.full(n, floor),
                stall_patience=jnp.full(n, patience), decay_factor=jnp.full(n, .5),
                max_queries=max_queries)


class IncumbentTests(unittest.TestCase):
    def test_pe_stall_restores_value_residual_and_cache(self):
        solver = d.WaterfallBESP_PE([[1.]], [[1.]], [1.], .5,
                                  h_init=8., h_min=1., stall_patience=2)
        for candidate in (1.5, 1., .5):
            with patch.object(solver, '_solve_khat', return_value=(jnp.array([candidate])-solver.v)/solver.h):
                solver.step()
        np.testing.assert_allclose(solver.V, [1.5])
        np.testing.assert_allclose(solver.br, [.25])
        self.assertEqual(solver.current_br_norm, .25)
        self.assertEqual(solver.history_br[-1], .25)
        self.assertEqual(solver.total_queries, 3)
        self.assertLess(solver.h, 8.)

    def test_pe_guard_restores_after_worsening_and_charges_once(self):
        for invalid in (1e12, float('nan'), float('inf'), RuntimeError('fixture')):
            with self.subTest(invalid=repr(invalid)):
                s = d.WaterfallBESP_PE([[1.]], [[1.]], [1.], .5, h_init=8., h_min=1., stall_patience=5)
                for x in (1.5, 1.):
                    with patch.object(s, '_solve_khat', return_value=(jnp.array([x])-s.v)/s.h):
                        s.step()
                args = dict(side_effect=invalid) if isinstance(invalid, Exception) else dict(return_value=(jnp.array([invalid])-s.v)/s.h)
                with patch.object(s, '_solve_khat', **args):
                    cost, resid = s.step()
                np.testing.assert_allclose(s.V, [1.5])
                np.testing.assert_allclose(s.br, [.25])
                self.assertEqual((cost, resid, s.total_queries), (1, .25, 3))

    def test_control_stall_and_guard_restore_cache_and_policy(self):
        for guard in (False, True):
            with self.subTest(guard=guard):
                s = control_solver(stall_patience=5 if guard else 2)
                control_step(s, 1.5, policy=1)
                control_step(s, 1., policy=0)
                _, resid, policy = control_step(s, 1e12 if guard else .5, policy=0)
                np.testing.assert_allclose(s.V, [1.5])
                np.testing.assert_allclose(s._last_expected_v_true, [[1.5, 1.5]])
                np.testing.assert_array_equal(policy, [1])
                self.assertAlmostEqual(resid, .25)
                self.assertAlmostEqual(s.history_br[-1], .25)
                self.assertEqual(s.total_queries, 3)

    def test_floor_and_unchanged_h_do_not_restore(self):
        for is_control in (False, True):
            for reason in ('floor', 'decay_one'):
                with self.subTest(control=is_control, reason=reason):
                    s = control_solver(stall_patience=1) if is_control else d.WaterfallBESP_PE([[1.]], [[1.]], [1.], .5, h_init=8., h_min=1., stall_patience=1)
                    if reason == 'floor':
                        s.h = s.h_min
                    else:
                        s.decay_factor = 1.
                    for x in (1.5, 1.):
                        if is_control:
                            control_step(s, x)
                        else:
                            with patch.object(s, '_solve_khat', return_value=(jnp.array([x])-s.v)/s.h):
                                s.step()
                    np.testing.assert_allclose(s.V, [1.])
                    self.assertAlmostEqual(s.current_br_norm, .5)
                    s._record_rejected_attempt()
                    np.testing.assert_allclose(s.V, [1.])
                    self.assertEqual(s.stop_due_to_failed_floor, reason == 'floor')

    def test_strict_tie_keeps_earlier_incumbent_and_floor_guard_restores(self):
        for is_control in (False, True):
            s = control_solver(stall_patience=1) if is_control else d.WaterfallBESP_PE([[1.]], [[1.]], [1.], .5, h_init=8., h_min=1., stall_patience=1)
            for x in (1.5, 2.5): # both have residual .25
                if is_control:
                    control_step(s, x)
                else:
                    with patch.object(s, '_solve_khat', return_value=(jnp.array([x])-s.v)/s.h):
                        s.step()
            np.testing.assert_allclose(s.V, [1.5])
            s.h = 2.
            s.decay_factor = .5
            s.v = jnp.array([1.])
            s._record_rejected_attempt()
            np.testing.assert_allclose(s.V, [1.5])
            self.assertTrue(s.stop_due_to_failed_floor)

    def test_batched_pe_fixture_parity_and_cached_next_rhs(self):
        for guard in (False, True):
            for at_floor in (False, True):
                with self.subTest(guard=guard, at_floor=at_floor):
                    values = [1.5, 1., 1e12 if guard else .5, 1.75]
                    hs = []
                    rhs = []
                    expected = [2.]
                    s = d.WaterfallBESP_PE([[1.]], [[1.]], [1.], .5, h_init=1. if at_floor else 8., h_min=1., stall_patience=2, decay_factor=.5)
                    for x in values:
                        hs.append(s.h)
                        with patch.object(s, '_solve_khat', return_value=(jnp.array([x])-s.v)/s.h):
                            s.step()
                        expected.append(abs(2-float(s.v[0])))
                        if s.stop_due_to_failed_floor:
                            expected.extend([expected[-1]]*(5-len(expected)))
                            break
                    index = 0
                    def solve(mat, br):
                        nonlocal index
                        rhs.append(np.asarray(br).copy())
                        old_value = (1-br[..., 0])/.5
                        result = ((values[index]-old_value)/hs[index])[..., None]
                        index += 1
                        return result
                    with patch.object(dispatch.jnp.linalg, 'solve', side_effect=solve):
                        q, errors = dispatch._batched_waterfall_pe_common(jnp.ones((1,1,1)), jnp.ones((1,1,1)), jnp.ones((1,1)), jnp.array([.5]), jnp.array([[2.]]), value_norm='inf', value_normalized=False, **batch_kwargs(h=1. if at_floor else 8.))
                    np.testing.assert_allclose(errors[0], expected)
                    np.testing.assert_array_equal(q[0], np.arange(5))
                    if not at_floor:
                        np.testing.assert_allclose(rhs[3], [[[.25]]])

    def test_batched_control_fixture_parity_and_cached_correction(self):
        for guard in (False, True):
            values = [1.5, 1., 1e12 if guard else .5, 1.75]
            s = control_solver()
            expected_r, expected_e = [], []
            for x in values:
                _, resid, _ = control_step(s, x)
                expected_r.append(resid)
                expected_e.append(abs(2-float(s.V[0])))
            index = 0
            def inner(v, p_hat, r_tilde, gamma_eff, tol, cap):
                nonlocal index
                if index == 3:
                    np.testing.assert_allclose(v, [[1.5]])
                    # P == Phat: coherent restored cached PV makes correction zero.
                    h = gamma_eff/(.5-gamma_eff)
                    expected_rtilde = v[...,None]/(1+h[:,None,None]) + h[:,None,None]*jnp.array([[[1., .9]]])/(1+h[:,None,None])
                    np.testing.assert_allclose(r_tilde, expected_rtilde)
                out = jnp.full_like(v, values[index])
                index += 1
                return out
            p = jnp.ones((1,1,2,1))
            common = dict(inner_vi_tol=jnp.array([1e-10]), inner_vi_max_iters=jnp.array([1]), **batch_kwargs())
            with patch.object(dispatch, '_batched_waterfall_inner_vi', side_effect=inner):
                q, residuals, errors, policy_errors, policies = dispatch._batched_waterfall_control_traces(p,p,jnp.array([[[1.,.9]]]),jnp.array([.5]),jnp.array([[2.]]),value_norm='inf',value_normalized=False,**common)
            np.testing.assert_allclose(residuals[0], expected_r)
            np.testing.assert_allclose(errors[0], expected_e)
            np.testing.assert_array_equal(policies, 0)
            np.testing.assert_allclose(policy_errors, 0)
            index = 0
            with patch.object(dispatch, '_batched_waterfall_inner_vi', side_effect=inner):
                hits = dispatch._batched_waterfall_control_hit_times(p,p,jnp.array([[[1.,.9]]]),jnp.array([.5]),jnp.array([.2]),**common)
            self.assertEqual(hits, [4])

    def test_direct_pe_linear_system_protocol(self):
        p = jnp.array([[.2,.8],[.7,.3]])
        s = d.WaterfallBESP_PE(p, jnp.eye(2), [1.,-.5], .9, h_init=3.)
        w = s._solve_khat(s.br, s.h)
        np.testing.assert_allclose(((1+s.h)*jnp.eye(2)-s.h*.9*s.p_hat_pi)@w, s.br, rtol=1e-13, atol=1e-13)
        self.assertEqual(w.dtype, jnp.float64)

    def test_initial_incumbent_and_linear_fallback(self):
        s = control_solver(stall_patience=1)
        _, resid, policy = control_step(s, -1., policy=1)
        np.testing.assert_array_equal(s.V, [0.])
        np.testing.assert_array_equal(s._last_expected_v_true, [[0.,0.]])
        self.assertEqual(resid, 1.)
        self.assertIsNone(policy)
        with patch.object(d.jnp.linalg, 'solve', side_effect=RuntimeError('fixture')):
            result = d._solve_linear(jnp.eye(2), jnp.array([2.,3.]))
        np.testing.assert_allclose(result, [2.,3.])

    def test_batched_control_mixed_floor_stop_and_policy_restore(self):
        p = jnp.ones((2,1,2,1))
        ph = jnp.array(p, copy=True)
        values = [[1.5,1.5],[1.,1.],[1e12,.5],[1.75,1.75]]
        index, model_calls, true_calls = 0, 0, 0
        real_einsum = jnp.einsum
        def einsum(spec, model, value):
            nonlocal model_calls, true_calls
            result = real_einsum(spec,model,value)
            if model is p:
                true_calls += 1
            if model is ph:
                model_calls += 1
                if model_calls % 2 == 0: # output policy at injected candidate
                    result = result.at[:,:,1 if index == 1 else 0].add(10.)
            return result
        def inner(v,*args):
            nonlocal index
            if index == 3:
                np.testing.assert_allclose(v, [[1.5],[1.5]])
            result = jnp.array(values[index])[:,None]
            index += 1
            return result
        with patch.object(dispatch,'_batched_waterfall_inner_vi',side_effect=inner), patch.object(dispatch.jnp,'einsum',side_effect=einsum):
            q, resid, errors, pe, policies = dispatch._batched_waterfall_control_traces(p,ph,jnp.array([[[1.,.9]],[[1.,.9]]]),jnp.array([.5,.5]),jnp.array([[2.],[2.]]),h_init=jnp.array([2.,8.]),h_min=jnp.array([1.,1.]),stall_patience=jnp.array([5,2]),decay_factor=jnp.array([.5,.5]),inner_vi_tol=jnp.array([1e-10,1e-10]),inner_vi_max_iters=jnp.array([1,1]),max_queries=4,value_norm='inf',value_normalized=False)
        np.testing.assert_allclose(resid[:,:3], [[.25,.5,.25],[.25,.5,.25]])
        np.testing.assert_array_equal(policies[:,:3,0], [[1,0,1],[1,0,1]])
        np.testing.assert_allclose(pe[:,:3], [[.2,0,.2],[.2,0,.2]], atol=1e-14)
        self.assertTrue(np.isinf(resid[0,3]))
        self.assertAlmostEqual(float(resid[1,3]),.125)
        # Two initialization products and one candidate true product per iteration;
        # the policy diagnostics have a distinct einsum signature/model selection.
        self.assertEqual(true_calls, 6)

    def test_control_finite_inner_protocol_and_scalar_batch_parity(self):
        # The stopping rule returns the PRE-update iterate on delta<tol.
        for tol, cap, expected in [(1.,5,0.),(.5,1,.5),(0.,2,.625),(0.,0,0.)]:
            s = d.WaterfallISP_Control([[[1.]]],[[[1.]]],[[1.]],.5,h_init=1.,inner_vi_tol=tol,inner_vi_max_iters=cap)
            s.step()
            self.assertEqual(float(s.V[0]), expected)
            out = dispatch._batched_waterfall_inner_vi(jnp.zeros((1,1)),jnp.ones((1,1,1,1)),jnp.array([[[.5]]]),jnp.array([.25]),jnp.array([tol]),jnp.array([cap]))
            np.testing.assert_allclose(out, [[expected]])

    def test_real_finite_control_and_pe_batch_parity(self):
        p = jnp.array([[.2,.8],[.7,.3]])
        ph = jnp.eye(2)
        reward = jnp.array([1.,-.5])
        target = jnp.linalg.solve(jnp.eye(2)-.9*p,reward)
        s = d.WaterfallBESP_PE(p,ph,reward,.9,h_init=32.,h_min=1.,stall_patience=2,decay_factor=.5)
        expected = [float(jnp.max(jnp.abs(target-s.V)))]
        for _ in range(12):
            s.step()
            expected.append(float(jnp.max(jnp.abs(target-s.V))))
        _, errors = dispatch._batched_waterfall_pe_common(p[None],ph[None],reward[None],jnp.array([.9]),target[None],value_norm='inf',value_normalized=False,**batch_kwargs(h=32.,max_queries=12))
        np.testing.assert_allclose(errors[0],expected,rtol=1e-12,atol=1e-12)
        ctrl = d.WaterfallISP_Control(p[:,None,:],ph[:,None,:],reward[:,None],.9,h_init=32.,h_min=1.,stall_patience=2,decay_factor=.5,inner_vi_tol=1e-10,inner_vi_max_iters=30)
        _, residuals, vals, policies = ctrl.run_with_policy_trace(12)
        _, br, be, _, bp = dispatch._batched_waterfall_control_traces(p[None,:,None,:],ph[None,:,None,:],reward[None,:,None],jnp.array([.9]),target[None],value_norm='inf',value_normalized=False,inner_vi_tol=jnp.array([1e-10]),inner_vi_max_iters=jnp.array([30]),**batch_kwargs(h=32.,max_queries=12))
        np.testing.assert_allclose(br[0],residuals,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(be[0],jnp.max(jnp.abs(vals-target),axis=1),rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(bp[0],policies)


if __name__ == '__main__':
    unittest.main(verbosity=2)
