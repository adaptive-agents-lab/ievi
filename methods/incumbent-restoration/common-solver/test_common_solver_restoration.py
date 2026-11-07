import unittest
from unittest.mock import patch
import numpy as np
import run_control_accuracy_common_solver as m
from control_accuracy_common import InnerSolveResult

class Cache:
 def __init__(self): self.policies=[]
 def error(self,policy): self.policies.append(policy.copy());return float(policy[0])

class IncumbentTests(unittest.TestCase):
 def run_case(self,values,budget=6):
  P=np.ones((1,2,1));Ph=P.copy();R=np.ones((1,2));cache=Cache();seen=[];actual=[0,0]
  einsum=np.einsum
  def counted_einsum(expr,operator,value):
   actual[0 if operator is P else 1]+=1
   return einsum(expr,operator,value)
  def solve(ph,rewards,beta,v,delta,**kw):
   i=len(seen);seen.append((v.copy(),kw['initial_phat_value'].copy(),rewards.copy(),beta))
   val=values[min(i,len(values)-1)]
   return InnerSolveResult(np.array([val]),np.array([1 if i==0 else 0],np.int32),2,1,0.,0.,True)
  with patch.object(m,'TRUE_BUDGET',budget),patch.object(m,'certified_surrogate_vi_control',solve),patch.object(m.np,'einsum',counted_einsum):
   out,att,c,status,_=m.run_waterfall(P,Ph,R,np.array([100.]),cache,1e-2)
  self.assertEqual(actual,[c.n_p,c.attempts])
  self.assertEqual(c.n_phat,actual[1]+c.attempts)  # one mocked inner product per solve
  return out,att,c,status,seen,cache
 def check_restart_input(self,seen,index):
  v,phv,reward,beta=seen[index]
  np.testing.assert_array_equal(v,[50.]);np.testing.assert_array_equal(phv,[[50.,50.]])
  h=beta/(m.GAMMA-beta)
  np.testing.assert_allclose(reward,50/(1+h)+h/(1+h))
 def test_stall_restores_compatible_tuple_without_queries(self):
  out,att,c,status,seen,cache=self.run_case([50.,0.,0.,0.,60.])
  self.check_restart_input(seen,4)
  self.assertTrue(att[3]['restored_incumbent']);self.assertEqual(att[3]['restoration_reason'],'stall')
  self.assertEqual(out[3]['true_bellman_residual'],.5);self.assertEqual(out[3]['policy_value_error'],1.)
  self.assertEqual(out[3]['candidate_true_bellman_residual'],1.)
  self.assertEqual((c.n_p,c.n_phat,c.attempts,c.accepted,c.rejected),(6,10,5,5,0))
 def test_guard_restores_compatible_tuple_without_queries(self):
  out,att,c,status,seen,cache=self.run_case([50.,0.,1e20,60.],budget=5)
  self.check_restart_input(seen,3)
  self.assertTrue(att[2]['restored_incumbent']);self.assertEqual(att[2]['restoration_reason'],'guard')
  self.assertEqual(att[2]['current_residual_after'],.5)
  self.assertEqual((c.n_p,c.n_phat,c.attempts,c.accepted,c.rejected),(5,10,5,4,1))
 def test_floor_guard_failure_and_no_charge(self):
  out,att,c,status,seen,cache=self.run_case([50.,1e20],budget=100)
  self.assertEqual(status,'failed_at_h_floor');self.assertEqual(att[-1]['h_after'],1.)
  self.assertEqual(sum(a['restored_incumbent'] for a in att),6)
  self.assertEqual((c.n_p,c.n_phat,c.attempts),(2,14,7))
 def test_floor_stall_does_not_restore_and_ties_do_not_replace(self):
  out,att,c,status,seen,cache=self.run_case([50.,0.],budget=24)
  self.assertEqual(sum(a['restored_incumbent'] for a in att),6)
  self.assertFalse(att[-1]['restored_incumbent']);self.assertEqual(att[-1]['h_after'],1.)
  self.assertEqual(out[-1]['true_bellman_residual'],1.)
  out,att,c,status,seen,cache=self.run_case([50.,50.,50.,50.,60.])
  self.assertEqual(att[3]['incumbent_attempt'],1);self.assertEqual(out[3]['policy_value_error'],1.)
 def test_inner_cap_does_not_trigger_step_reduction(self):
  P=np.ones((1,1,1));R=np.ones((1,1))
  result=InnerSolveResult(np.array([0.]),np.array([0],np.int32),5000,4999,1.,100.,False)
  with patch.object(m,'certified_surrogate_vi_control',return_value=result):
   out,att,c,status,_=m.run_waterfall(P,P,R,np.array([100.]),Cache(),1e-2)
  self.assertEqual(status,'inner_solver_cap');self.assertEqual((c.n_p,c.n_phat),(1,5000))
  self.assertFalse(att[0]['restored_incumbent'])

if __name__=='__main__': unittest.main(verbosity=2)
