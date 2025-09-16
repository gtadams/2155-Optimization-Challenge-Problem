import jax.numpy as np
from ._utils import equisample, find_optimal_correspondences
import jax
from typing import List, Tuple, Optional, Union

class CurveEngine:
    def __init__(self,
                 resolution : Optional[int] = 200,
                 equisample : Optional[bool] = True,
                 normalize_scale : Optional[bool] = True,
                 device : Optional[Union[jax.Device, str]] = 'cpu'):
        
        self.resolution = resolution
        self.equisample = equisample
        self.normalize_scale = normalize_scale
        
        if isinstance(device, jax.Device):
            self.device = device
        elif device=='cpu':
            self.device = jax.devices('cpu')[0]
        elif device=='gpu':
            self.device = jax.devices('gpu')[0]
        else:
            raise ValueError("Device must be 'cpu' or 'gpu' or a jax.Device instance.")
    
    def compile(self):
        self.__call__ = jax.jit(self.__call__, device=self.device)
        self.optimal_alignment = jax.jit(self.optimal_alignment, device=self.device)
        self.compare_curves = jax.jit(self.compare_curves, device=self.device)
    
    def __call__(self, curves: np.ndarray):
        '''
        A class to process and standardize curves for comparison and visualization.
        
        Parameters
        ----------
        curves : np.ndarray
            An array of curves, where each curve is represented as a sequence of 2D points.
        
        Returns
        -------
        np.ndarray
            Processed curves with standardized resolution and scale.
        '''
        with jax.default_device(self.device):
            if self.equisample:
                curves = equisample(curves, self.resolution)
            
            n = curves.shape[1]
            
            # center curves
            curves = curves - curves.mean(1)[:,None]
            
            if self.normalize_scale:
                # apply uniform scaling
                s = ((np.square(curves).sum(-1).sum(-1)/n)**0.5)[:,None,None]
                curves = curves/s
            
            return curves
    
    def optimal_alignment(self, 
                          curves: np.ndarray, 
                          target_curves: np.ndarray,
                          return_normalized: bool = True,
                          return_distances: bool = True,
                          return_transforms: bool = False):
        '''
        Aligns a set of curves to a target curve using optimal rotation and translation.
        
        Parameters
        ----------
        curves : np.ndarray
            An array of curves to be aligned.
        target_curve : np.ndarray
            The target curve to which the input curves will be aligned.
        return_normalized : bool, optional
            Whether to return the normalized curves. Default is True. If False, returns the aligned curves exactly aligned to the input target curves (without normalization or equisampling).
        return_distances : bool, optional
            Whether to return the distances between each aligned curve and the target curve. Default is True. This distance will always be computed in the normalized space (equisampled and scaled if indicated at initialization).
        return_transforms : bool, optional
            Whether to return the transformation parameters (rotation, translation, and scale) used for alignment.
        
        Returns
        -------
        aligned_curves : np.ndarray
            The aligned curves. If return_normalized is True, these will be the normalized curves, otherwise they will be the input curves aligned to the input target curves.
        aligned_target_curves : np.ndarray
            The normalized target curves if return_normalized is True, otherwise the input target curves.
        distances : np.ndarray, optional
            The distances between each aligned curve and the target curve. Returned only if return_distances is True.
        transforms : Tuple(T, R, S), optional
            A tuple of transformation parameters for each curve. T is the translation vector, R is the rotation matrix, and S is the scale factor. Shaped (num_curves, 2), (num_curves, 2, 2), and (num_curves, 1) respectively. Returned only if return_transforms is True.
        '''
        with jax.default_device(self.device):
            single_input = False
            if target_curves.ndim == 2 and curves.ndim == 2:
                target_curves = target_curves[None]
                curves = curves[None]
                single_input = True
            elif target_curves.ndim == 2:
                target_curves = target_curves[None]
                target_curves = target_curves.repeat(curves.shape[0], axis=0)
            elif curves.ndim == 2:
                raise ValueError("The number of curves must match the number of target curves if multiple target curves are provided.")
            
            # First apply relavent normalizations
            if self.equisample:
                normalized_curves = equisample(curves, self.resolution)
                normalized_targets = equisample(target_curves, self.resolution)
            elif curves.shape[1] != target_curves.shape[1]:
                raise ValueError("If equisampling is disabled, the number of points in the input curves must match the number of points in the target curves.")
            else:
                normalized_curves = curves.copy()
                normalized_targets = target_curves.copy()
            
            curves_centroid = normalized_curves.mean(1)[:,None]
            targets_centroid = normalized_targets.mean(1)[:,None]
            normalized_curves = normalized_curves - curves_centroid
            normalized_targets = normalized_targets - targets_centroid

            n = normalized_curves.shape[1]
            
            if self.normalize_scale:
                s = ((np.square(normalized_curves).sum(-1).sum(-1)/n)**0.5)[:,None,None]
                normalized_curves = normalized_curves/s
                
                s_target = ((np.square(normalized_targets).sum(-1).sum(-1)/n)**0.5)[:,None,None]
                normalized_targets = normalized_targets/s_target

            optimal_correspondences, R, aligned_curves, distances = find_optimal_correspondences(normalized_curves,
                                                                                                normalized_targets,
                                                                                                return_rotation_matrices=True,
                                                                                                return_aligned_curves=True,
                                                                                                return_distances=True)

            T = targets_centroid - curves_centroid
            if self.normalize_scale:
                S = (s_target/s).reshape(-1,1)
            else:
                S = np.ones((curves.shape[0], 1))
            
            if return_normalized:
                aligned_targets = normalized_targets
            else:
                aligned_curves = np.einsum('bij,bkj->bki', R, curves - curves_centroid) * S[:,None] + targets_centroid
                aligned_targets = target_curves
            
            if single_input:
                outputs = (aligned_curves[0], aligned_targets[0])
                distances = distances[0]
                T = T[0]
                R = R[0]
                S = S[0]
            else:
                outputs = (aligned_curves, aligned_targets)

            if return_distances:
                outputs = outputs + (distances,)
            
            if return_transforms:
                outputs = outputs + ((T, R, S),)
                
            return outputs
    
    def compare_curves(self,
                       curves: np.ndarray,
                       target_curves: np.ndarray):
        '''
        Compares a set of curves to a target curve using optimal alignment.
        Parameters
        ----------
        curves : np.ndarray
            An array of curves to be compared.  Shape (num_curves, num_points, 2).
        target_curves : np.ndarray
            The target curve to which the input curves will be compared. Shape (num_target_curves, num_points, 2).
        Returns
        -------
        distances : np.ndarray
            The distances between each input curve and the target curve after optimal alignment. Shape (num_curves, num_target_curves).
        '''
        with jax.default_device(self.device):
            aligned_curves, aligned_targets, distances = self.optimal_alignment(curves, target_curves,
                                                                                return_normalized=True,
                                                                                return_distances=True,
                                                                                return_transforms=False)
            
            return distances