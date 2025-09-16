import jax.numpy as jnp
import numpy as np
import jax
from typing import Optional, Union, List
from ._utils import *
from ..Kinematics._MechanismSolver import MechanismSolver
from ..Kinematics._kin import sort_mechanism

def build_optimization_functions(x0, edges, fixed_joints, motor, target_curve):
    N = len(x0)
    edges, fixed_joints, N, motor, x0, order, mapping = sort_mechanism(x0, edges, fixed_joints, N, motor)
    
    solver_instance = MechanismSolver(max_size=x0.shape[0], timesteps=target_curve.shape[0], is_sorted=False)

    As_, x0s_, node_types_, thetas_, target_idx = solver_instance.create_batch_from_list(
        x0s=[x0],
        edges=[edges],
        fixed_nodes=[fixed_joints]
    )
    thetas_ = np.linspace(0, 2 * np.pi, solver_instance.timesteps)

    optimizer_target = equisample(target_curve[None], solver_instance.timesteps)
    
    def objective_and_gradient(x0):
        
        x0__ = x0[order].reshape(x0s_.shape)
        
        vals, grad = differentiated_solve_and_assess_od(
            x0__,
            As_,
            node_types_,
            thetas_,
            target_idx,
            optimizer_target,
            1.0
        )
        
        obj = vals[0]
        if np.any(np.isnan(grad)):
           obj = np.inf

        return obj, grad[0][mapping].reshape(x0.shape)

    return objective_and_gradient

def build_scaled_optimization_functions(x0, edges, fixed_joints, motor, target_curve, scale: float):
    N = len(x0)
    edges, fixed_joints, N, motor, x0, order, mapping = sort_mechanism(x0, edges, fixed_joints, N, motor)
    
    solver_instance = MechanismSolver(max_size=x0.shape[0], timesteps=target_curve.shape[0], is_sorted=False)

    As_, x0s_, node_types_, thetas_, target_idx = solver_instance.create_batch_from_list(
        x0s=[x0],
        edges=[edges],
        fixed_nodes=[fixed_joints]
    )
    thetas_ = np.linspace(0, 2 * np.pi, solver_instance.timesteps)

    optimizer_target = equisample(target_curve[None], solver_instance.timesteps)
    
    def objective_and_gradient(x0):
        
        x0__ = x0[order].reshape(x0s_.shape)
        
        vals, grad = differentiated_solve_and_assess_od_scaled(
            x0__,
            As_,
            node_types_,
            thetas_,
            target_idx,
            optimizer_target,
            scale,
            1.0
        )
        
        obj = vals[0]
        if np.any(np.isnan(grad)):
           obj = np.inf

        return obj, grad[0][mapping].reshape(x0.shape)

    return objective_and_gradient

def build_material_optimization_functions(edges):
    
    def material(x0):
        lengths = jnp.linalg.norm(x0[edges[:,0]] - x0[edges[:,1]], axis=1)
        obj = jnp.sum(lengths)
        
        return obj
    
    return jax.jit(jax.value_and_grad(material))