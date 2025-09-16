import jax.numpy as jnp
import numpy as np
import jax
from ._PathSynthesis import PathSynthesis
from tqdm.auto import tqdm, trange
from typing import Union, Optional, Callable

class BatchBFGS:
    def __init__(self, 
                 max_iter: Optional[int] = 100,
                 max_iter_line_search: Optional[int] = 10,
                 stop_threshhold: Optional[float] = 0.01,
                 tol: Optional[float] = 1e-6,
                 c1: Optional[float] = 1e-4, 
                 c2: Optional[float] = 0.9,
                 verbose: Optional[bool] = False,
                 device: Optional[Union[str, jax.Device]] = 'cpu',
                 alpha_decay: Optional[Callable] = None,
                 base_alpha: Optional[float] = 1.0):
        
        self.max_iter = max_iter
        self.max_iter_line_search = max_iter_line_search
        self.stop_threshhold = stop_threshhold
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        self.base_alpha = base_alpha
        self.alpha_decay = alpha_decay if alpha_decay is not None else lambda x: 1.0
        self.current_alpha = base_alpha
        
        if isinstance(device, jax.Device):
            self.device = device
        
        elif device=='cpu':
            self.device = jax.devices('cpu')[0]
        elif device=='gpu':
            self.device = jax.devices('gpu')[0]
        
        self.initialized = False
    
    def initialize(self, problem: PathSynthesis):
        if problem.device != self.device:
            raise ValueError(f"Problem device {problem.device} does not match BatchBFGS device {self.device}.")
        
        with jax.default_device(self.device):
            if not problem.initialized:
                raise ValueError("Problem must be initialized before optimization.")
            
            self.desvars = np.array(problem.desvars)
            self.problem = problem
            self.batch_size = self.desvars.shape[0]
            self.dim = self.desvars.shape[1]

            self.H = np.eye(self.dim)[None].repeat(self.batch_size, axis=0)

            self.f = np.array(problem.f)
            self.nabla = np.array(problem.df)
            self.p = -(self.H @ self.nabla[...,None]).squeeze(-1)

            self.s = np.zeros_like(self.desvars)

            self.initialized = True
    
    def solve(self):
        if not self.initialized:
            raise ValueError("BatchBFGS not initialized. Call initialize() first.")
        
        if self.verbose:
            pbar = trange(self.max_iter, desc="Solving ...")
        else:
            pbar = range(self.max_iter)
        
        status = -1
        for i in pbar:
            self.current_alpha = self.alpha_decay(i) * self.base_alpha
            self.iterate()
            convergence = np.linalg.norm(self.s, axis=-1, ord=np.inf) < self.tol
            
            if self.verbose:
                pbar.set_postfix(
                    {"Mean Objective": np.mean(self.f[~np.isnan(self.f)]),
                     "Best Objective": np.min(self.f[~np.isnan(self.f)])}
                )

            if np.all(convergence):
                status = 1
                break

            if np.any(self.problem.f < self.stop_threshhold):
                status = 2
                break
        
        return status
        
    def iterate(self):
        with jax.default_device(self.device):
            alphas = np.ones((self.batch_size, 1)) * self.current_alpha
            
            for _ in range(self.max_iter_line_search):
                new_desvars = self.desvars + alphas * self.p
                self.problem.set_desvars(new_desvars)
                f_new = self.problem.f
                nabla_new = self.problem.df

                armijo = f_new - self.f - (self.c1 * alphas * np.sum(self.nabla * self.p, axis=-1, keepdims=True)).squeeze(-1)
                curvature = -np.sum(nabla_new * self.p, axis=-1) + self.c2 * np.sum(self.nabla * self.p, axis=-1)

                conditions_met = (armijo <= 0) & (curvature <= 0)

                if np.all(conditions_met):
                    break
                
                alphas[~conditions_met] *= 0.5
                # alphas = jnp.where(conditions_met.reshape(*alphas.shape), alphas, alphas * 0.5)
            
            old_desvars = np.copy(self.desvars)
            # self.desvars = self.desvars * (~conditions_met)[:,None] + new_desvars * conditions_met[:,None]
            self.desvars[conditions_met] = new_desvars[conditions_met]
            
            if not np.all(conditions_met):
                not_met_but_improved = np.logical_and(~conditions_met, f_new <= self.f)
                # self.desvars = self.desvars * (~not_met_but_improved)[:,None] + new_desvars * not_met_but_improved[:,None]
                self.desvars[not_met_but_improved] = new_desvars[not_met_but_improved]
                
            self.problem.set_desvars(self.desvars)
            
            # check if numerically stable
            broken = np.isinf(self.problem.f)
            if np.any(broken):
                # self.desvars = self.desvars * (~broken)[:, None] + old_desvars * broken[:, None]
                self.desvars[broken] = old_desvars[broken]
                self.problem.set_desvars(self.desvars)
            
            nabla_new = np.array(self.problem.df)
            
            y = nabla_new - self.nabla
            self.s = self.desvars - old_desvars

            sdoty = np.sum(self.s * y, axis=-1, keepdims=True)[..., None]
            sst = self.s[:, :, None] @ self.s[:, None, :]
            yst = y[:, :, None] @ self.s[:, None, :]
            syt = self.s[:, :, None] @ y[:, None, :]
            y_inner = (y[:, None, :] @ self.H @ y[:, :, None])
            
            self.H = self.H + (sdoty + y_inner)/np.square(sdoty) * sst - (self.H @ yst + syt @ self.H) / sdoty

            self.f = np.array(self.problem.f)
            self.nabla = np.array(self.problem.df)
            self.p = -(self.H @ self.nabla[..., None]).squeeze(-1)