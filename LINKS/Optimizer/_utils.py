import jax
import jax.numpy as jnp
from ..Kinematics._solvers import JAX_Solve

def equisample(curves, n):

    l = jnp.cumsum(jnp.pad(jnp.linalg.norm(curves[:,1:,:] - curves[:,:-1,:],axis=-1),((0,0),(1,0))),axis=-1)
    l = l/l[:,-1].reshape(-1,1)

    sampling = jnp.linspace(-1e-6,1-1e-6,n)
    end_is = jax.vmap(lambda a: jnp.searchsorted(a.reshape(-1),sampling)[1:])(l)

    end_ids = end_is

    l_end = l[jnp.arange(end_is.shape[0]).reshape(-1,1),end_is]
    l_start = l[jnp.arange(end_is.shape[0]).reshape(-1,1),end_is-1]
    ws = (l_end - sampling[1:].reshape(1,-1))/(l_end-l_start)

    end_gather = curves[jnp.arange(end_ids.shape[0]).reshape(-1,1),end_ids]
    start_gather = curves[jnp.arange(end_ids.shape[0]).reshape(-1,1),end_ids-1]

    uniform_curves = jnp.concatenate([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws[:,:,None])],1)

    return uniform_curves

def normalize(curves):
    
    n = curves.shape[1]
    # center curves
    curves = curves - curves.mean(1)[:,None]
    
    # apply uniform scaling
    s = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1)/n)[:,None,None]
    curves = curves/s
    
    return curves

@jax.jit
def _euclidean_distance(x, y) -> float:
    dist = jax.numpy.sqrt(jax.numpy.sum((x - y) ** 2))
    return dist

def cdist(a, b):
    return jax.vmap(lambda x,y: jax.vmap(lambda x1: jax.vmap(lambda y1: _euclidean_distance(x1, y1))(y))(x))(a,b)

def batch_chamfer_distance(c1,c2):

    d = cdist(c1,c2)
    id1 = d.argmin(1)
    id2 = d.argmin(2)

    d1 = jax.numpy.linalg.norm(c2 - c1[jax.numpy.arange(id1.shape[0]).reshape(-1,1),id1],axis=-1).mean(1)
    d2 = jax.numpy.linalg.norm(c1 - c2[jax.numpy.arange(id2.shape[0]).reshape(-1,1),id2],axis=-1).mean(1)

    return d1 + d2

def align_optimal(curves, target_curves, normalize_curves=True, n=200):
    if normalize_curves:
        normalized_candidates = normalize(equisample(curves, n))
        normalized_target = normalize(equisample(target_curves, n))
    else:
        normalized_candidates = curves
        normalized_target = target_curves

    
    n = normalized_target.shape[1]
    init_order = jax.numpy.arange(n)

    base_order_set = init_order[init_order[:,None]-jax.numpy.zeros_like(init_order)].T
    base_order_set = base_order_set.repeat(2, axis=0)
    permuted_order_set = init_order[init_order[:,None]-init_order].T 

    permuted_order_set = jax.numpy.concatenate([permuted_order_set, jax.numpy.copy(permuted_order_set[:,::-1])], axis=0)
    
    permuted_targets = normalized_target[
        jnp.arange(normalized_target.shape[0]).repeat(base_order_set.shape[0]).reshape(-1,1),
        base_order_set[None].repeat(normalized_target.shape[0], axis=0).reshape(-1,base_order_set.shape[-1])
    ].reshape(-1, base_order_set.shape[0], n, 2)
    
    permuted_candidates = normalized_candidates[
        jnp.arange(normalized_candidates.shape[0]).repeat(permuted_order_set.shape[0]).reshape(-1,1),
        permuted_order_set[None].repeat(normalized_candidates.shape[0], axis=0).reshape(-1,permuted_order_set.shape[-1])
    ].reshape(-1, permuted_order_set.shape[0], n, 2)
    
    dots = (permuted_targets * permuted_candidates).sum([2,3])
    vars = (permuted_targets[...,1]*permuted_candidates[...,0] - permuted_targets[...,0]*permuted_candidates[...,1]).sum(-1)
    thetas = jnp.arctan2(vars, dots)
    rotation_matrices = jnp.stack([jnp.cos(thetas), -jnp.sin(thetas), jnp.sin(thetas), jnp.cos(thetas)], axis=-1).reshape(-1, permuted_order_set.shape[0], 2, 2)

    rotated_candidates = (rotation_matrices @ permuted_candidates.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

    d_od = jnp.linalg.norm(permuted_targets - rotated_candidates, axis=-1).sum(-1)/n * 2 * jnp.pi

    best_match = jnp.argmin(d_od, axis=-1)

    aligned_candidates = rotated_candidates[jnp.arange(rotated_candidates.shape[0]), best_match]

    return aligned_candidates, normalized_target, d_od.min(axis=-1)

def align_optimal_scaled(curves, target_curves, scale, normalize_curves=True, n=200):
    
    if normalize_curves:
        normalized_candidates = normalize(equisample(curves, n)) * scale
        normalized_target = normalize(equisample(target_curves, n))
        
        o_scale = jax.numpy.sqrt(jax.numpy.square(curves).sum(-1).sum(-1)/n)[:,None,None]
        normalized_candidates *= o_scale
    else:
        normalized_candidates = curves * scale
        normalized_target = target_curves

    
    n = normalized_target.shape[1]
    init_order = jax.numpy.arange(n)

    base_order_set = init_order[init_order[:,None]-jax.numpy.zeros_like(init_order)].T
    base_order_set = base_order_set.repeat(2, axis=0)
    permuted_order_set = init_order[init_order[:,None]-init_order].T 

    permuted_order_set = jax.numpy.concatenate([permuted_order_set, jax.numpy.copy(permuted_order_set[:,::-1])], axis=0)
    
    permuted_targets = normalized_target[
        jnp.arange(normalized_target.shape[0]).repeat(base_order_set.shape[0]).reshape(-1,1),
        base_order_set[None].repeat(normalized_target.shape[0], axis=0).reshape(-1,base_order_set.shape[-1])
    ].reshape(-1, base_order_set.shape[0], n, 2)
    
    permuted_candidates = normalized_candidates[
        jnp.arange(normalized_candidates.shape[0]).repeat(permuted_order_set.shape[0]).reshape(-1,1),
        permuted_order_set[None].repeat(normalized_candidates.shape[0], axis=0).reshape(-1,permuted_order_set.shape[-1])
    ].reshape(-1, permuted_order_set.shape[0], n, 2)
    
    dots = (permuted_targets * permuted_candidates).sum([2,3])
    vars = (permuted_targets[...,1]*permuted_candidates[...,0] - permuted_targets[...,0]*permuted_candidates[...,1]).sum(-1)
    thetas = jnp.arctan2(vars, dots)
    rotation_matrices = jnp.stack([jnp.cos(thetas), -jnp.sin(thetas), jnp.sin(thetas), jnp.cos(thetas)], axis=-1).reshape(-1, permuted_order_set.shape[0], 2, 2)

    rotated_candidates = (rotation_matrices @ permuted_candidates.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

    d_od = jnp.linalg.norm(permuted_targets - rotated_candidates, axis=-1).sum(-1)/n * 2 * jnp.pi

    best_match = jnp.argmin(d_od, axis=-1)

    aligned_candidates = rotated_candidates[jnp.arange(rotated_candidates.shape[0]), best_match]

    return aligned_candidates, normalized_target, d_od.min(axis=-1)

def chamfer_distance(curves, target_curves, normalize_curves=True, n=200):
    if normalize_curves:
        normalized_candidates = normalize(equisample(curves, n))
        normalized_target = normalize(equisample(target_curves, n))
    else:
        normalized_candidates = curves
        normalized_target = target_curves
        
    chamfer_distance = batch_chamfer_distance(normalized_candidates, normalized_target)
    
    return chamfer_distance

def singularity_check(x0s_, As_, node_types_, thetas_):
    worst_offender = JAX_Solve(
        As_, x0s_, node_types_, thetas_
    )[1]
    
    return worst_offender.sum(), worst_offender

def solve_and_assess(x0s_, As_, node_types_, thetas_, target_idx, target_curves, weight_chamfer, weight_distance):
    solutions = JAX_Solve(
        As_,x0s_,node_types_,thetas_
    )[0]
    
    n = thetas_.shape[0]
    solved_candidates = solutions[jnp.arange(solutions.shape[0]), target_idx]
    
    has_nans = jnp.isnan(solved_candidates).any(axis=-1).any(axis=-1)
    solved_candidates_safe = jnp.nan_to_num(solved_candidates * (~has_nans)[:,None,None], nan=0.0) + target_curves * has_nans[:,None,None]
    
    optimal_solved_candidates, normalized_target, d_od = align_optimal(solved_candidates_safe, target_curves, n=n)
    cds = chamfer_distance(optimal_solved_candidates, normalized_target, normalize_curves=False, n=n)

    obj = d_od.sum() * weight_distance + cds.sum() * weight_chamfer
    d_od = (d_od) * (~has_nans) + has_nans * jnp.inf
    
    return obj, (d_od, optimal_solved_candidates, normalized_target)

def solve_and_assess_od(x0s_, As_, node_types_, thetas_, target_idx, target_curves, weight_distance):
    solutions = JAX_Solve(
        As_,x0s_,node_types_,thetas_
    )[0]
    
    n = thetas_.shape[0]
    solved_candidates = solutions[jnp.arange(solutions.shape[0]), target_idx]
    
    has_nans = jnp.isnan(solved_candidates).any(axis=-1).any(axis=-1)
    solved_candidates_safe = jnp.nan_to_num(solved_candidates * (~has_nans)[:,None,None], nan=0.0) + target_curves * has_nans[:,None,None]
    
    optimal_solved_candidates, normalized_target, d_od = align_optimal(solved_candidates_safe, target_curves, n=n)

    obj = d_od.sum() * weight_distance
    d_od = d_od * (~has_nans) + has_nans * jnp.inf
    
    return obj, (d_od, optimal_solved_candidates, normalized_target)

def solve_and_assess_od_scaled(x0s_, As_, node_types_, thetas_, target_idx, target_curves, scale, weight_distance):
    solutions = JAX_Solve(
        As_,x0s_,node_types_,thetas_
    )[0]
    
    n = thetas_.shape[0]
    solved_candidates = solutions[jnp.arange(solutions.shape[0]), target_idx]
    
    has_nans = jnp.isnan(solved_candidates).any(axis=-1).any(axis=-1)
    solved_candidates_safe = jnp.nan_to_num(solved_candidates * (~has_nans)[:,None,None], nan=0.0) + target_curves * has_nans[:,None,None]

    optimal_solved_candidates, normalized_target, d_od = align_optimal_scaled(solved_candidates_safe, target_curves, scale, n=n)

    obj = d_od.sum() * weight_distance
    d_od = d_od * (~has_nans) + has_nans * jnp.inf
    
    return obj, (d_od, optimal_solved_candidates, normalized_target)

differentiated_solve_and_assess = jax.jit(jax.value_and_grad(solve_and_assess,has_aux=True))
differentiated_solve_and_assess_od = jax.jit(jax.value_and_grad(solve_and_assess_od, has_aux=True))
differentiated_singularity_check = jax.jit(jax.value_and_grad(singularity_check, has_aux=True))
differentiated_solve_and_assess_od_scaled = jax.jit(jax.value_and_grad(solve_and_assess_od_scaled, has_aux=True))