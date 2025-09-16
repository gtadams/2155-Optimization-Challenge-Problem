import jax.numpy as jnp
import numpy as np
import jax
from typing import Optional, Union, List
from ._utils import *
from ..Kinematics._MechanismSolver import MechanismSolver

class PathSynthesis:
    def __init__(self, 
                 max_size: int = 20,
                 num_timesteps: int = 1000,
                 device: Union[str, jax.Device] = 'cpu',
                 use_chamfer_distance: bool = False,
                 weight_chamfer: float = 1.0,
                 weight_distance: float = 1.0):
        
        self.max_size = max_size
        self.curve_resolution = num_timesteps
        self.timesteps = num_timesteps
        self.use_chamfer_distance = use_chamfer_distance
        self.target_path = None
        self.has_target = False
        self.initialized = False
        self.weight_chamfer = weight_chamfer
        self.weight_distance = weight_distance/2/np.pi
        
        if isinstance(device, jax.Device):
            self.device = device
        
        elif device=='cpu':
            self.device = jax.devices('cpu')[0]
        elif device=='gpu':
            self.device = jax.devices('gpu')[0]
        
        self.solver = MechanismSolver(max_size=max_size, timesteps=num_timesteps, is_sorted=True, device=self.device)
        
    def set_target_path(self,
                        target_path: Union[np.ndarray, jnp.ndarray]):
        self.target_path = jnp.array(target_path)[None]
        self.has_target = True
        
    def init_desvars(self,
                     x0s: List[np.ndarray],
                     edges: List[np.ndarray],
                     fixed_nodes: List[np.ndarray],
                     start_theta: float = 0.0,
                     end_theta: float = 2 * np.pi):
        
        if not self.has_target:
            raise ValueError("Target path not set. Call set_target_path() first.")
        
        As_, x0s_, node_types_, thetas_, target_idx = self.solver.create_batch_from_list(
            x0s=x0s,
            edges=edges,
            fixed_nodes=fixed_nodes
        )
        thetas_ = np.linspace(start_theta, end_theta, self.timesteps)
        
        self.args_passable = [x0s_, As_, node_types_, thetas_, target_idx]
        
        self.optimizer_target = equisample(self.target_path.repeat(len(x0s_), axis=0), self.curve_resolution)
        
        self.initialized = True
        
        self.solve()
    
    def set_desvars(self, desvars: jnp.ndarray):
        if not self.initialized:
            raise ValueError("PathSynthesis not initialized. Call init_desvars() first.")
        
        self.args_passable[0] = desvars.reshape(*self.args_passable[0].shape)
        self.solve()
            
    def solve(self):
        if not self.initialized:
            raise ValueError("PathSynthesis not initialized. Call init_desvars() first.")
        if not self.has_target:
            raise ValueError("Target path not set. Call set_target_path() first.")

        if self.use_chamfer_distance:
            vals, grad = differentiated_solve_and_assess(
                *self.args_passable,
                self.optimizer_target,
                self.weight_chamfer,
                self.weight_distance
            )
        else:
            vals, grad = differentiated_solve_and_assess_od(
                *self.args_passable,
                self.optimizer_target,
                self.weight_distance
            )

        self.f = vals[1][0]
        self.aux = vals[1]
        self.df = jnp.nan_to_num(grad, nan=0.0).reshape(grad.shape[0], -1)
        
        
    @property
    def desvars(self):
        return self.args_passable[0].reshape(self.args_passable[0].shape[0], -1)
    
    def get_best_solution(self):
        if not self.initialized:
            raise ValueError("PathSynthesis not initialized. Call init_desvars() first.")
        
        best_idx = jnp.argmin(self.f)
        x0_best = self.args_passable[0][best_idx]
        A_best = self.args_passable[1][best_idx]
        node_types_best = self.args_passable[2][best_idx]
        size = self.args_passable[4][best_idx] + 1

        edges = np.vstack(np.where(np.tril(A_best[:size,:][:, :size]))).T

        traced_curve = np.array(self.aux[1][best_idx])
        
        mechanism = {}
        mechanism['initial positions'] = x0_best[:size].tolist()
        mechanism['edges'] = edges.tolist()
        mechanism['fixed joints'] = np.where(node_types_best[:size])[0].tolist()
        mechanism['target path'] = traced_curve
        
        return mechanism
        