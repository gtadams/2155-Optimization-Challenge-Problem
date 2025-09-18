from ..Optimization import Tools
import numpy as np
from typing import List, Union
from pymoo.indicators.hv import HV

def make_empty_submission():
    return {
        'Problem 1': [],
        'Problem 2': [],
        'Problem 3': [],
        'Problem 4': [],
        'Problem 5': [],
        'Problem 6': []
    }

def evaluate_submission(
    submission: Union[dict, str],
    target_curves: Union[np.ndarray, str] = 'target_curves.npy') -> float:
    
    optimization_tools = Tools(
        timesteps=200,
        max_size=20,
        material=True,
        scaled=False,
        device='cpu'
    )
    optimization_tools.compile()
    
    ref_point = np.array([0.75, 10.0])
    ind = HV(ref_point)
    
    if isinstance(submission, str):
        submission = np.load(submission, allow_pickle=True).item()
    if isinstance(target_curves, str):
        target_curves = np.load(target_curves)
        
    scores = []
    for problem in range(6):
        problem_key = f'Problem {problem + 1}'
        if problem_key in submission:
            if len(submission[problem_key]) == 0:
                scores.append(0.0)
                continue
            
            x0s = []
            edges = []
            fixed_joints = []
            motors = []
            target_idx = []
            
            counter = 0
            for item in submission[problem_key]:
                counter += 1
                if counter > 1000:
                    print(f"Warning: More than 1000 designs submitted for {problem_key}. Only the first 1000 will be evaluated.")
                    break
                if 'x0' not in item or 'edges' not in item or 'fixed_joints' not in item or 'motor' not in item:
                    # Invalid entry, skip
                    continue
                
                x0s.append(np.array(item['x0']))
                edges.append(np.array(item['edges']))
                fixed_joints.append(np.array(item['fixed_joints']))
                motors.append(np.array(item['motor']))
                target_idx.append(item.get('target_joint', None))
            
            if len(x0s) > 0:
                distances, material = optimization_tools(
                    x0s=x0s,
                    edges=edges,
                    fixed_joints=fixed_joints,
                    motors=motors,
                    target_curve=target_curves[problem],
                    target_idx=target_idx
                )
            else:
                scores.append(0.0)
                continue
                
            F = np.vstack([distances, material]).T
            valids = np.logical_and(F[:,0] < 0.75, F[:,1] < 10.0)
            if np.sum(valids) == 0:
                scores.append(0.0)
                continue
            
            score = ind(F[valids])
            
            scores.append(score)
            
    return {'Overall Score': float(np.mean(scores)), 'Score Breakdown': {
        f'Problem {i + 1}': float(scores[i]) for i in range(6)
    }}