"""
Dynamic Belief Parameters System for Time-Varying Distributions

This module provides mathematically rigorous interpolation between belief distribution
parameters over time, enabling the modeling of crisis-driven belief evolution.

Mathematical Foundation:
- Parameter interpolation using multiple algorithms (linear, cubic spline, sigmoid)
- Smooth transitions preserving distribution properties
- Crisis scenario modeling with realistic temporal dynamics
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize_scalar
import json

from .continuous_beliefs import BeliefDistributionParams, DistributionType


class InterpolationMethod(Enum):
    """Supported interpolation methods for parameter transitions"""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"
    STEP = "step"


class CrisisType(Enum):
    """Types of crisis scenarios for belief evolution modeling"""
    PANDEMIC = "pandemic"
    ELECTION = "election"
    ECONOMIC_SHOCK = "economic_shock"
    NATURAL_DISASTER = "natural_disaster"
    SOCIAL_UNREST = "social_unrest"
    TECHNOLOGICAL_DISRUPTION = "technological_disruption"


@dataclass
class ParameterKeyframe:
    """
    A keyframe defining belief parameters at a specific time point.
    
    Mathematical Properties:
    - Contains complete parameter state at time t
    - Supports interpolation weight specification
    - Maintains parameter validity constraints
    """
    
    time_point: int                                    # Round number
    parameters: BeliefDistributionParams               # Parameter state
    interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR
    transition_speed: float = 1.0                     # Speed modifier for transitions
    confidence: float = 1.0                           # Confidence in this keyframe (0-1)


@dataclass
class DynamicBeliefParameters:
    """
    Time-varying belief distribution parameters with mathematical interpolation.
    
    Mathematical Foundation:
    - Supports multiple interpolation algorithms: P(t) = φ(P₀, P₁, ..., Pₙ, t)
    - Preserves parameter constraints during interpolation
    - Enables crisis scenario modeling with realistic dynamics
    - Provides derivative information for rate-of-change analysis
    """
    
    # Core timeline definition
    keyframes: List[ParameterKeyframe] = field(default_factory=list)
    
    # Global interpolation settings
    default_interpolation: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE
    extrapolation_mode: str = "constant"  # 'constant', 'linear', 'error'
    
    # Mathematical properties
    smoothing_factor: float = 0.0         # Smoothing parameter for splines
    boundary_conditions: str = "natural"   # 'natural', 'clamped', 'periodic'
    
    # Validation settings
    validate_parameters: bool = True
    parameter_bounds_checking: bool = True
    
    def __post_init__(self):
        """Initialize and validate the dynamic parameter system"""
        if not self.keyframes:
            raise ValueError("At least one keyframe must be provided")
        
        # Sort keyframes by time
        self.keyframes.sort(key=lambda k: k.time_point)
        
        # Validate keyframes
        if self.validate_parameters:
            self._validate_keyframes()
        
        # Build interpolation functions
        self._build_interpolators()
    
    def _validate_keyframes(self):
        """Validate mathematical consistency of keyframes"""
        for i, keyframe in enumerate(self.keyframes):
            # Check time points are non-negative and unique
            if keyframe.time_point < 0:
                raise ValueError(f"Keyframe {i}: time_point must be non-negative")
            
            # Check for duplicate time points
            time_points = [k.time_point for k in self.keyframes]
            if len(set(time_points)) != len(time_points):
                raise ValueError("Keyframes must have unique time points")
            
            # Validate parameter ranges
            params = keyframe.parameters
            if self.parameter_bounds_checking:
                if not (-1.0 <= params.center <= 1.0):
                    warnings.warn(f"Keyframe {i}: center {params.center} outside [-1, 1]")
                if not (0.0 <= params.polarization_strength <= 1.0):
                    warnings.warn(f"Keyframe {i}: polarization_strength {params.polarization_strength} outside [0, 1]")
                if not (-1.0 <= params.polarization_asymmetry <= 1.0):
                    warnings.warn(f"Keyframe {i}: polarization_asymmetry {params.polarization_asymmetry} outside [-1, 1]")
    
    def _build_interpolators(self):
        """Build interpolation functions for each parameter"""
        if len(self.keyframes) < 2:
            # Single keyframe - constant parameters
            self._interpolators = {}
            return
        
        time_points = np.array([k.time_point for k in self.keyframes])
        
        # Extract parameter values
        parameter_series = {
            'polarization_strength': [k.parameters.polarization_strength for k in self.keyframes],
            'polarization_asymmetry': [k.parameters.polarization_asymmetry for k in self.keyframes],
            'gap_size': [k.parameters.gap_size for k in self.keyframes],
            'center': [k.parameters.center for k in self.keyframes],
            'spread': [k.parameters.spread for k in self.keyframes],
            'skewness': [k.parameters.skewness for k in self.keyframes],
            'concentration': [k.parameters.concentration for k in self.keyframes],
            'noise_level': [k.parameters.noise_level for k in self.keyframes]
        }
        
        # Build interpolators for each parameter
        self._interpolators = {}
        
        for param_name, values in parameter_series.items():
            values_array = np.array(values)
            
            # Choose interpolation method
            method = self.default_interpolation
            
            if method == InterpolationMethod.LINEAR:
                self._interpolators[param_name] = interp1d(
                    time_points, values_array, 
                    kind='linear', 
                    bounds_error=False,
                    fill_value=(values_array[0], values_array[-1])
                )
            
            elif method == InterpolationMethod.CUBIC_SPLINE:
                # Use scipy's CubicSpline for smooth interpolation
                self._interpolators[param_name] = CubicSpline(
                    time_points, values_array,
                    bc_type=self.boundary_conditions
                )
            
            elif method == InterpolationMethod.SIGMOID:
                # Custom sigmoid interpolation for smooth transitions
                self._interpolators[param_name] = self._create_sigmoid_interpolator(
                    time_points, values_array
                )
            
            else:
                # Default to linear
                self._interpolators[param_name] = interp1d(
                    time_points, values_array, 
                    kind='linear', 
                    bounds_error=False,
                    fill_value=(values_array[0], values_array[-1])
                )
    
    def _create_sigmoid_interpolator(self, time_points: np.ndarray, values: np.ndarray) -> Callable:
        """Create a sigmoid-based interpolator for smooth transitions"""
        
        def sigmoid_interp(t):
            """Sigmoid interpolation between keyframes"""
            if len(time_points) == 1:
                return values[0]
            
            # Find surrounding keyframes
            if t <= time_points[0]:
                return values[0]
            if t >= time_points[-1]:
                return values[-1]
            
            # Find interval
            idx = np.searchsorted(time_points, t) - 1
            t0, t1 = time_points[idx], time_points[idx + 1]
            v0, v1 = values[idx], values[idx + 1]
            
            # Sigmoid interpolation
            progress = (t - t0) / (t1 - t0)
            sigmoid_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            
            return v0 + (v1 - v0) * sigmoid_progress
        
        return sigmoid_interp
    
    def get_parameters_at_time(self, time_point: float) -> BeliefDistributionParams:
        """
        Get interpolated parameters at a specific time point.
        
        Mathematical Process:
        1. Check if exact keyframe exists
        2. Apply interpolation algorithm
        3. Validate parameter constraints
        4. Return new BeliefDistributionParams object
        
        Args:
            time_point: Time to get parameters for (can be fractional)
            
        Returns:
            Interpolated BeliefDistributionParams object
        """
        
        # Handle single keyframe case
        if len(self.keyframes) == 1:
            return self.keyframes[0].parameters
        
        # Check for exact keyframe match
        for keyframe in self.keyframes:
            if abs(keyframe.time_point - time_point) < 1e-10:
                return keyframe.parameters
        
        # Handle extrapolation
        if time_point < self.keyframes[0].time_point:
            if self.extrapolation_mode == "constant":
                return self.keyframes[0].parameters
            elif self.extrapolation_mode == "error":
                raise ValueError(f"Time point {time_point} before first keyframe")
        
        if time_point > self.keyframes[-1].time_point:
            if self.extrapolation_mode == "constant":
                return self.keyframes[-1].parameters
            elif self.extrapolation_mode == "error":
                raise ValueError(f"Time point {time_point} after last keyframe")
        
        # Interpolate parameters
        interpolated_values = {}
        
        for param_name, interpolator in self._interpolators.items():
            try:
                interpolated_values[param_name] = float(interpolator(time_point))
            except Exception as e:
                warnings.warn(f"Interpolation failed for {param_name}: {e}")
                # Fallback to linear interpolation
                interpolated_values[param_name] = self._fallback_linear_interpolation(
                    param_name, time_point
                )
        
        # Create new parameters object
        base_params = self.keyframes[0].parameters
        
        return replace(base_params, **interpolated_values)
    
    def _fallback_linear_interpolation(self, param_name: str, time_point: float) -> float:
        """Fallback linear interpolation when main interpolation fails"""
        
        time_points = [k.time_point for k in self.keyframes]
        values = [getattr(k.parameters, param_name) for k in self.keyframes]
        
        # Find surrounding points
        idx = np.searchsorted(time_points, time_point) - 1
        if idx < 0:
            return values[0]
        if idx >= len(values) - 1:
            return values[-1]
        
        # Linear interpolation
        t0, t1 = time_points[idx], time_points[idx + 1]
        v0, v1 = values[idx], values[idx + 1]
        
        alpha = (time_point - t0) / (t1 - t0)
        return v0 + alpha * (v1 - v0)
    
    def get_parameter_derivative(self, time_point: float, param_name: str, 
                               epsilon: float = 0.01) -> float:
        """
        Compute the derivative (rate of change) of a parameter at a specific time.
        
        Mathematical Foundation:
        Uses finite difference approximation: f'(t) ≈ [f(t+ε) - f(t-ε)] / (2ε)
        
        Args:
            time_point: Time to compute derivative at
            param_name: Parameter name to differentiate
            epsilon: Step size for finite difference
            
        Returns:
            Rate of change of the parameter at time_point
        """
        
        if param_name not in self._interpolators:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        # Central difference approximation
        try:
            params_before = self.get_parameters_at_time(time_point - epsilon)
            params_after = self.get_parameters_at_time(time_point + epsilon)
            
            value_before = getattr(params_before, param_name)
            value_after = getattr(params_after, param_name)
            
            derivative = (value_after - value_before) / (2 * epsilon)
            return derivative
            
        except Exception as e:
            warnings.warn(f"Derivative computation failed: {e}")
            return 0.0
    
    def detect_rapid_changes(self, param_name: str, threshold: float = 0.1) -> List[Tuple[float, float]]:
        """
        Detect periods of rapid parameter change (phase transitions).
        
        Mathematical Method:
        1. Sample parameter values at regular intervals
        2. Compute derivatives using finite differences  
        3. Identify points where |derivative| > threshold
        4. Group consecutive rapid-change points into periods
        
        Args:
            param_name: Parameter to analyze
            threshold: Minimum derivative magnitude to consider "rapid"
            
        Returns:
            List of (start_time, end_time) tuples for rapid change periods
        """
        
        if len(self.keyframes) < 2:
            return []
        
        # Sample parameter values
        start_time = self.keyframes[0].time_point
        end_time = self.keyframes[-1].time_point
        sample_points = np.linspace(start_time, end_time, 100)
        
        # Compute derivatives
        derivatives = []
        for t in sample_points:
            try:
                deriv = self.get_parameter_derivative(t, param_name)
                derivatives.append(abs(deriv))
            except:
                derivatives.append(0.0)
        
        derivatives = np.array(derivatives)
        
        # Find rapid change periods
        rapid_change_mask = derivatives > threshold
        rapid_periods = []
        
        in_rapid_period = False
        period_start = None
        
        for i, is_rapid in enumerate(rapid_change_mask):
            if is_rapid and not in_rapid_period:
                # Start of rapid period
                in_rapid_period = True
                period_start = sample_points[i]
            elif not is_rapid and in_rapid_period:
                # End of rapid period
                in_rapid_period = False
                rapid_periods.append((period_start, sample_points[i-1]))
        
        # Handle case where rapid period extends to end
        if in_rapid_period:
            rapid_periods.append((period_start, sample_points[-1]))
        
        return rapid_periods
    
    def get_timeline_summary(self) -> Dict:
        """
        Generate a comprehensive summary of the dynamic parameter timeline.
        
        Returns:
            Dictionary containing timeline statistics and analysis
        """
        
        if not self.keyframes:
            return {}
        
        # Basic timeline info
        summary = {
            'num_keyframes': len(self.keyframes),
            'time_span': self.keyframes[-1].time_point - self.keyframes[0].time_point,
            'start_time': self.keyframes[0].time_point,
            'end_time': self.keyframes[-1].time_point,
            'interpolation_method': self.default_interpolation.value
        }
        
        # Parameter range analysis
        param_ranges = {}
        param_names = ['polarization_strength', 'polarization_asymmetry', 'gap_size', 
                      'center', 'spread', 'skewness', 'concentration']
        
        for param in param_names:
            values = [getattr(k.parameters, param) for k in self.keyframes]
            param_ranges[param] = {
                'min': min(values),
                'max': max(values), 
                'range': max(values) - min(values),
                'initial': values[0],
                'final': values[-1],
                'total_change': values[-1] - values[0]
            }
        
        summary['parameter_ranges'] = param_ranges
        
        # Rapid change analysis
        rapid_changes = {}
        for param in param_names:
            rapid_periods = self.detect_rapid_changes(param)
            rapid_changes[param] = {
                'num_rapid_periods': len(rapid_periods),
                'rapid_periods': rapid_periods
            }
        
        summary['rapid_changes'] = rapid_changes
        
        return summary
    
    def save_to_file(self, filepath: str):
        """Save dynamic parameters to JSON file"""
        
        # Convert to serializable format
        data = {
            'keyframes': [],
            'default_interpolation': self.default_interpolation.value,
            'extrapolation_mode': self.extrapolation_mode,
            'smoothing_factor': self.smoothing_factor,
            'boundary_conditions': self.boundary_conditions
        }
        
        for kf in self.keyframes:
            keyframe_data = {
                'time_point': kf.time_point,
                'interpolation_method': kf.interpolation_method.value,
                'transition_speed': kf.transition_speed,
                'confidence': kf.confidence,
                'parameters': {
                    'distribution_type': kf.parameters.distribution_type.value,
                    'center': kf.parameters.center,
                    'spread': kf.parameters.spread,
                    'skewness': kf.parameters.skewness,
                    'polarization_strength': kf.parameters.polarization_strength,
                    'polarization_asymmetry': kf.parameters.polarization_asymmetry,
                    'gap_size': kf.parameters.gap_size,
                    'concentration': kf.parameters.concentration,
                    'noise_level': kf.parameters.noise_level,
                    'random_seed': kf.parameters.random_seed
                }
            }
            data['keyframes'].append(keyframe_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DynamicBeliefParameters':
        """Load dynamic parameters from JSON file"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct keyframes
        keyframes = []
        for kf_data in data['keyframes']:
            params_data = kf_data['parameters']
            
            parameters = BeliefDistributionParams(
                distribution_type=DistributionType(params_data['distribution_type']),
                center=params_data['center'],
                spread=params_data['spread'],
                skewness=params_data['skewness'],
                polarization_strength=params_data['polarization_strength'],
                polarization_asymmetry=params_data['polarization_asymmetry'],
                gap_size=params_data['gap_size'],
                concentration=params_data['concentration'],
                noise_level=params_data['noise_level'],
                random_seed=params_data.get('random_seed')
            )
            
            keyframe = ParameterKeyframe(
                time_point=kf_data['time_point'],
                parameters=parameters,
                interpolation_method=InterpolationMethod(kf_data['interpolation_method']),
                transition_speed=kf_data['transition_speed'],
                confidence=kf_data['confidence']
            )
            keyframes.append(keyframe)
        
        # Create instance
        return cls(
            keyframes=keyframes,
            default_interpolation=InterpolationMethod(data['default_interpolation']),
            extrapolation_mode=data['extrapolation_mode'],
            smoothing_factor=data['smoothing_factor'],
            boundary_conditions=data['boundary_conditions']
        )


class CrisisScenarioGenerator:
    """
    Generate realistic crisis scenarios for belief evolution studies.
    
    Mathematical Foundation:
    - Based on empirical studies of crisis-driven opinion dynamics
    - Parameterized models for different crisis types
    - Configurable severity, duration, and recovery patterns
    """
    
    @staticmethod
    def pandemic_scenario(severity: float = 0.8, duration_rounds: int = 25) -> DynamicBeliefParameters:
        """
        Generate a pandemic crisis scenario.
        
        Mathematical Model:
        - Initial uncertainty (moderate polarization)
        - Peak polarization during crisis peak
        - Gradual recovery with persistent elevation
        
        Args:
            severity: Crisis severity (0.0 to 1.0)
            duration_rounds: Total duration in simulation rounds
            
        Returns:
            DynamicBeliefParameters for pandemic scenario
        """
        
        # Scale parameters by severity
        peak_polarization = 0.3 + 0.6 * severity
        uncertainty_phase_pol = 0.1 + 0.3 * severity
        recovery_pol = 0.2 + 0.4 * severity
        
        # Calculate unique time points for keyframes
        if duration_rounds < 10:
            # For short experiments, use simple progression
            time_points = [0, max(1, duration_rounds // 4), max(2, duration_rounds // 2), 
                          max(3, 3 * duration_rounds // 4), duration_rounds]
        else:
            # For longer experiments, use the original formula
            time_points = [0, max(3, duration_rounds // 8), duration_rounds // 3, 
                          2 * duration_rounds // 3, duration_rounds]
        
        # Ensure all time points are unique and sorted
        time_points = sorted(list(set(time_points)))
        
        # If we don't have enough unique points, interpolate
        while len(time_points) < 5:
            # Add a point between existing points
            for i in range(len(time_points) - 1):
                mid_point = (time_points[i] + time_points[i + 1]) // 2
                if mid_point not in time_points and mid_point > time_points[i]:
                    time_points.insert(i + 1, mid_point)
                    break
        
        keyframes = [
            # Pre-crisis: moderate beliefs
            ParameterKeyframe(
                time_point=time_points[0],
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.NORMAL,
                    center=0.0,
                    spread=0.4,
                    polarization_strength=0.1,
                    noise_level=0.05
                )
            ),
            
            # Initial uncertainty
            ParameterKeyframe(
                time_point=time_points[1],
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=0.0,
                    polarization_strength=uncertainty_phase_pol,
                    polarization_asymmetry=0.1,  # Slight bias
                    gap_size=0.2,
                    noise_level=0.1
                )
            ),
            
            # Peak crisis polarization
            ParameterKeyframe(
                time_point=time_points[2],
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=0.0,
                    polarization_strength=peak_polarization,
                    polarization_asymmetry=0.2,  # Stronger bias
                    gap_size=0.4,
                    noise_level=0.15
                )
            ),
            
            # Adaptation phase
            ParameterKeyframe(
                time_point=time_points[3],
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=0.0,
                    polarization_strength=recovery_pol,
                    polarization_asymmetry=0.15,
                    gap_size=0.3,
                    noise_level=0.1
                )
            ),
            
            # New normal
            ParameterKeyframe(
                time_point=time_points[4],
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.NORMAL,
                    center=0.1,  # Slight shift
                    spread=0.5,
                    polarization_strength=0.3,  # Elevated baseline
                    noise_level=0.08
                )
            )
        ]
        
        return DynamicBeliefParameters(
            keyframes=keyframes,
            default_interpolation=InterpolationMethod.CUBIC_SPLINE
        )
    
    @staticmethod
    def election_scenario(polarization_peak: float = 0.9, cycle_rounds: int = 20) -> DynamicBeliefParameters:
        """
        Generate an election cycle scenario.
        
        Mathematical Model:
        - Gradual polarization increase during campaign
        - Peak polarization before election
        - Sharp recovery post-election
        - Return to moderate baseline
        
        Args:
            polarization_peak: Maximum polarization level (0.0 to 1.0)
            cycle_rounds: Election cycle duration in rounds
            
        Returns:
            DynamicBeliefParameters for election scenario
        """
        
        keyframes = [
            # Early campaign
            ParameterKeyframe(
                time_point=0,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.NORMAL,
                    center=0.0,
                    spread=0.3,
                    polarization_strength=0.2,
                    noise_level=0.05
                )
            ),
            
            # Campaign heats up
            ParameterKeyframe(
                time_point=cycle_rounds // 4,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=0.0,
                    polarization_strength=0.5,
                    polarization_asymmetry=0.0,
                    gap_size=0.3,
                    noise_level=0.08
                )
            ),
            
            # Pre-election peak
            ParameterKeyframe(
                time_point=cycle_rounds // 2,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=0.0,
                    polarization_strength=polarization_peak,
                    polarization_asymmetry=0.1,
                    gap_size=0.5,
                    noise_level=0.12
                )
            ),
            
            # Election outcome (slight asymmetry from winner)
            ParameterKeyframe(
                time_point=cycle_rounds // 2 + 2,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=0.1,  # Shift toward winner
                    polarization_strength=0.7,
                    polarization_asymmetry=0.3,
                    gap_size=0.4,
                    noise_level=0.1
                )
            ),
            
            # Post-election normalization
            ParameterKeyframe(
                time_point=cycle_rounds,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.NORMAL,
                    center=0.05,  # Slight persistent bias
                    spread=0.4,
                    polarization_strength=0.3,
                    noise_level=0.06
                )
            )
        ]
        
        return DynamicBeliefParameters(
            keyframes=keyframes,
            default_interpolation=InterpolationMethod.SIGMOID
        )
    
    @staticmethod
    def economic_shock_scenario(shock_severity: float = 0.7, recovery_time: int = 30) -> DynamicBeliefParameters:
        """
        Generate an economic shock scenario.
        
        Mathematical Model:
        - Sudden shock creates immediate polarization
        - Prolonged period of high uncertainty
        - Gradual recovery with changed baseline
        
        Args:
            shock_severity: Economic shock severity (0.0 to 1.0)
            recovery_time: Recovery duration in rounds
            
        Returns:
            DynamicBeliefParameters for economic shock scenario
        """
        
        shock_polarization = 0.4 + 0.5 * shock_severity
        uncertainty_level = 0.1 + 0.1 * shock_severity
        
        keyframes = [
            # Pre-shock stable
            ParameterKeyframe(
                time_point=0,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.NORMAL,
                    center=0.0,
                    spread=0.35,
                    polarization_strength=0.15,
                    noise_level=0.03
                )
            ),
            
            # Immediate shock response
            ParameterKeyframe(
                time_point=2,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=-0.1,  # Negative bias during crisis
                    polarization_strength=shock_polarization,
                    polarization_asymmetry=-0.2,  # Left-skewed fear
                    gap_size=0.3,
                    noise_level=uncertainty_level
                ),
                interpolation_method=InterpolationMethod.STEP  # Sudden change
            ),
            
            # Sustained uncertainty
            ParameterKeyframe(
                time_point=recovery_time // 3,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=-0.05,
                    polarization_strength=shock_polarization + 0.1,
                    polarization_asymmetry=-0.15,
                    gap_size=0.35,
                    noise_level=uncertainty_level
                )
            ),
            
            # Recovery begins
            ParameterKeyframe(
                time_point=2 * recovery_time // 3,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.BIMODAL,
                    center=0.0,
                    polarization_strength=0.4,
                    polarization_asymmetry=-0.05,
                    gap_size=0.25,
                    noise_level=0.08
                )
            ),
            
            # New equilibrium
            ParameterKeyframe(
                time_point=recovery_time,
                parameters=BeliefDistributionParams(
                    distribution_type=DistributionType.NORMAL,
                    center=0.0,
                    spread=0.45,  # Increased spread post-crisis
                    polarization_strength=0.25,  # Elevated baseline
                    noise_level=0.05
                )
            )
        ]
        
        return DynamicBeliefParameters(
            keyframes=keyframes,
            default_interpolation=InterpolationMethod.CUBIC_SPLINE
        )
    
    @staticmethod
    def custom_scenario(crisis_type: CrisisType, 
                       severity: float = 0.5,
                       duration: int = 20,
                       asymmetry_bias: float = 0.0) -> DynamicBeliefParameters:
        """
        Generate a custom crisis scenario with specified parameters.
        
        Args:
            crisis_type: Type of crisis to model
            severity: Crisis severity (0.0 to 1.0)
            duration: Duration in simulation rounds
            asymmetry_bias: Population bias direction (-1.0 to 1.0)
            
        Returns:
            DynamicBeliefParameters for custom scenario
        """
        
        if crisis_type == CrisisType.PANDEMIC:
            return CrisisScenarioGenerator.pandemic_scenario(severity, duration)
        elif crisis_type == CrisisType.ELECTION:
            return CrisisScenarioGenerator.election_scenario(0.5 + 0.4 * severity, duration)
        elif crisis_type == CrisisType.ECONOMIC_SHOCK:
            return CrisisScenarioGenerator.economic_shock_scenario(severity, duration)
        else:
            # Generic crisis scenario
            peak_pol = 0.3 + 0.6 * severity
            
            keyframes = [
                ParameterKeyframe(
                    time_point=0,
                    parameters=BeliefDistributionParams(
                        distribution_type=DistributionType.NORMAL,
                        center=asymmetry_bias * 0.1,
                        spread=0.4,
                        polarization_strength=0.2,
                        noise_level=0.05
                    )
                ),
                ParameterKeyframe(
                    time_point=duration // 3,
                    parameters=BeliefDistributionParams(
                        distribution_type=DistributionType.BIMODAL,
                        center=asymmetry_bias * 0.15,
                        polarization_strength=peak_pol,
                        polarization_asymmetry=asymmetry_bias * 0.3,
                        gap_size=0.4,
                        noise_level=0.1
                    )
                ),
                ParameterKeyframe(
                    time_point=duration,
                    parameters=BeliefDistributionParams(
                        distribution_type=DistributionType.NORMAL,
                        center=asymmetry_bias * 0.05,
                        spread=0.45,
                        polarization_strength=0.3,
                        noise_level=0.06
                    )
                )
            ]
            
            return DynamicBeliefParameters(keyframes=keyframes)