"""
Advanced Mathematical Analysis for Dynamic Belief Evolution

This module provides sophisticated mathematical tools for analyzing belief
trajectories, detecting phase transitions, and modeling crisis dynamics.

Mathematical Foundation:
- Trajectory fitting using multiple model families (polynomial, exponential, logistic)
- Change point detection using CUSUM and Bayesian methods
- Fourier analysis for periodic patterns
- Information-theoretic measures for complexity analysis
- Statistical tests for significant changes
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats, signal, optimize
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import jarque_bera, normaltest, kstest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.regime_switching import markov_regression
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available, some advanced analysis features disabled")


class ModelType(Enum):
    """Supported trajectory model types"""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"
    SIGMOID = "sigmoid"
    PIECEWISE_LINEAR = "piecewise_linear"
    SPLINE = "spline"


class ChangePointMethod(Enum):
    """Methods for change point detection"""
    CUSUM = "cusum"
    BAYESIAN = "bayesian"
    VARIANCE_CHANGE = "variance_change"
    REGRESSION_TREE = "regression_tree"


@dataclass
class TrajectoryModel:
    """
    Mathematical model fitted to a belief trajectory.
    
    Contains model parameters, goodness-of-fit metrics, and
    mathematical properties of the trajectory.
    """
    
    model_type: ModelType
    parameters: Dict[str, float]
    r_squared: float
    aic: float                          # Akaike Information Criterion
    bic: float                          # Bayesian Information Criterion
    rmse: float                         # Root Mean Square Error
    
    # Mathematical properties
    turning_points: List[Tuple[float, float]] = field(default_factory=list)  # (time, value)
    inflection_points: List[Tuple[float, float]] = field(default_factory=list)
    monotonicity: str = "unknown"       # "increasing", "decreasing", "non-monotonic"
    
    # Model function
    predict_function: Optional[Callable] = None
    
    def predict(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict trajectory values at given time points"""
        if self.predict_function is None:
            raise ValueError("Model has no prediction function")
        return self.predict_function(t)
    
    def predict_derivative(self, t: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        """Compute trajectory derivatives"""
        if self.predict_function is None:
            raise ValueError("Model has no prediction function")
        
        # Numerical differentiation using finite differences
        epsilon = 0.01
        
        if order == 1:
            return (self.predict_function(t + epsilon) - self.predict_function(t - epsilon)) / (2 * epsilon)
        elif order == 2:
            return (self.predict_function(t + epsilon) - 2 * self.predict_function(t) + 
                   self.predict_function(t - epsilon)) / (epsilon ** 2)
        else:
            raise ValueError(f"Derivative order {order} not supported")


@dataclass
class ChangePoint:
    """
    Detected change point in a time series.
    
    Mathematical properties include location, magnitude,
    and statistical significance of the change.
    """
    
    location: float                     # Time of change point
    magnitude: float                    # Size of change
    confidence: float                   # Statistical confidence (0-1)
    change_type: str                    # "mean", "variance", "trend"
    
    # Statistical properties
    p_value: float                      # Statistical significance
    test_statistic: float               # Test statistic value
    
    # Context
    before_mean: float                  # Mean before change
    after_mean: float                   # Mean after change
    before_variance: float              # Variance before change  
    after_variance: float               # Variance after change


class TrajectoryAnalyzer:
    """
    Advanced trajectory analysis with multiple mathematical approaches.
    
    Mathematical Capabilities:
    - Multi-model trajectory fitting and selection
    - Change point detection using statistical tests
    - Fourier analysis for periodic patterns
    - Complexity measures and entropy analysis
    - Cross-trajectory correlation and synchronization
    """
    
    def __init__(self, smoothing_factor: float = 0.1, confidence_level: float = 0.95):
        self.smoothing_factor = smoothing_factor
        self.confidence_level = confidence_level
        
        # Analysis cache
        self._fitted_models = {}
        self._change_points = {}
        
    def fit_trajectory_models(self, time_points: np.ndarray, 
                            values: np.ndarray,
                            models: List[ModelType] = None) -> Dict[ModelType, TrajectoryModel]:
        """
        Fit multiple mathematical models to a trajectory.
        
        Mathematical Process:
        1. Fit candidate models using least squares
        2. Compute goodness-of-fit metrics (R², AIC, BIC)
        3. Analyze mathematical properties (turning points, monotonicity)
        4. Return ranked models by information criteria
        
        Args:
            time_points: Time coordinates
            values: Trajectory values
            models: List of model types to fit (if None, fits all)
            
        Returns:
            Dictionary mapping model types to fitted TrajectoryModel objects
        """
        
        if models is None:
            models = [ModelType.LINEAR, ModelType.QUADRATIC, ModelType.CUBIC, 
                     ModelType.EXPONENTIAL, ModelType.LOGISTIC, ModelType.SPLINE]
        
        fitted_models = {}
        
        for model_type in models:
            try:
                model = self._fit_single_model(time_points, values, model_type)
                fitted_models[model_type] = model
            except Exception as e:
                warnings.warn(f"Failed to fit {model_type.value} model: {e}")
        
        return fitted_models
    
    def _fit_single_model(self, t: np.ndarray, y: np.ndarray, model_type: ModelType) -> TrajectoryModel:
        """Fit a single model type to trajectory data"""
        
        n = len(y)
        
        if model_type == ModelType.LINEAR:
            # Linear regression: y = a*t + b
            coeffs = np.polyfit(t, y, 1)
            a, b = coeffs
            
            def predict_func(x):
                return a * x + b
            
            parameters = {'slope': a, 'intercept': b}
            monotonicity = "increasing" if a > 0 else "decreasing" if a < 0 else "constant"
            turning_points = []
            inflection_points = []
            
        elif model_type == ModelType.QUADRATIC:
            # Quadratic regression: y = a*t² + b*t + c
            coeffs = np.polyfit(t, y, 2)
            a, b, c = coeffs
            
            def predict_func(x):
                return a * x**2 + b * x + c
            
            parameters = {'a': a, 'b': b, 'c': c}
            
            # Find turning point (vertex)
            if abs(a) > 1e-10:
                vertex_t = -b / (2 * a)
                vertex_y = predict_func(vertex_t)
                turning_points = [(vertex_t, vertex_y)]
                monotonicity = "non-monotonic" if min(t) <= vertex_t <= max(t) else \
                              ("increasing" if a > 0 else "decreasing")
            else:
                turning_points = []
                monotonicity = "increasing" if b > 0 else "decreasing"
            
            inflection_points = []
            
        elif model_type == ModelType.CUBIC:
            # Cubic regression: y = a*t³ + b*t² + c*t + d
            coeffs = np.polyfit(t, y, 3)
            a, b, c, d = coeffs
            
            def predict_func(x):
                return a * x**3 + b * x**2 + c * x + d
            
            parameters = {'a': a, 'b': b, 'c': c, 'd': d}
            
            # Find critical points (roots of derivative)
            turning_points = []
            inflection_points = []
            
            if abs(a) > 1e-10:
                # Derivative: 3a*t² + 2b*t + c = 0
                discriminant = (2*b)**2 - 4*(3*a)*c
                if discriminant >= 0:
                    t1 = (-2*b + np.sqrt(discriminant)) / (2*3*a)
                    t2 = (-2*b - np.sqrt(discriminant)) / (2*3*a)
                    for tp in [t1, t2]:
                        if min(t) <= tp <= max(t):
                            turning_points.append((tp, predict_func(tp)))
                
                # Inflection point: 6a*t + 2b = 0
                inflection_t = -2*b / (6*a)
                if min(t) <= inflection_t <= max(t):
                    inflection_points.append((inflection_t, predict_func(inflection_t)))
            
            monotonicity = "non-monotonic" if turning_points else "monotonic"
            
        elif model_type == ModelType.EXPONENTIAL:
            # Exponential model: y = a * exp(b*t) + c
            try:
                # Use non-linear least squares
                def exp_func(x, a, b, c):
                    return a * np.exp(b * x) + c
                
                # Initial guess
                p0 = [1.0, 0.1, np.mean(y)]
                
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(exp_func, t, y, p0=p0, maxfev=1000)
                a, b, c = popt
                
                def predict_func(x):
                    return exp_func(x, a, b, c)
                
                parameters = {'a': a, 'b': b, 'c': c}
                monotonicity = "increasing" if a*b > 0 else "decreasing" if a*b < 0 else "constant"
                turning_points = []
                inflection_points = []
                
            except:
                # Fallback to log-linear if possible
                y_pos = y - np.min(y) + 1e-6
                log_y = np.log(y_pos)
                coeffs = np.polyfit(t, log_y, 1)
                a, b = coeffs
                
                def predict_func(x):
                    return np.exp(a * x + b)
                
                parameters = {'log_slope': a, 'log_intercept': b}
                monotonicity = "increasing" if a > 0 else "decreasing"
                turning_points = []
                inflection_points = []
        
        elif model_type == ModelType.LOGISTIC:
            # Logistic model: y = L / (1 + exp(-k*(t - t0)))
            try:
                def logistic_func(x, L, k, t0):
                    return L / (1 + np.exp(-k * (x - t0)))
                
                # Initial guess
                L = np.max(y) - np.min(y)
                t0 = np.median(t)
                k = 1.0
                
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(logistic_func, t, y, p0=[L, k, t0], maxfev=1000)
                L, k, t0 = popt
                
                def predict_func(x):
                    return logistic_func(x, L, k, t0)
                
                parameters = {'L': L, 'k': k, 't0': t0}
                
                # Inflection point is at t0
                inflection_points = [(t0, predict_func(t0))]
                turning_points = []
                monotonicity = "increasing" if k > 0 else "decreasing"
                
            except:
                # Fallback to sigmoid approximation
                def predict_func(x):
                    return np.mean(y) + 0 * x
                
                parameters = {'failed': True}
                monotonicity = "constant"
                turning_points = []
                inflection_points = []
        
        elif model_type == ModelType.SPLINE:
            # Smoothing spline
            spline = UnivariateSpline(t, y, s=self.smoothing_factor * len(y))
            
            def predict_func(x):
                return spline(x)
            
            parameters = {'smoothing_factor': self.smoothing_factor}
            
            # Find approximate turning points
            t_fine = np.linspace(min(t), max(t), 100)
            y_fine = spline(t_fine)
            dy_fine = spline.derivative()(t_fine)
            
            # Find where derivative changes sign
            sign_changes = np.where(np.diff(np.sign(dy_fine)))[0]
            turning_points = [(t_fine[i], y_fine[i]) for i in sign_changes]
            
            monotonicity = "non-monotonic" if turning_points else "monotonic"
            inflection_points = []
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Compute goodness-of-fit metrics
        y_pred = predict_func(t)
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE
        rmse = np.sqrt(ss_res / n)
        
        # AIC and BIC (approximate for non-linear models)
        k = len(parameters)  # Number of parameters
        log_likelihood = -n/2 * np.log(ss_res/n)
        aic = 2*k - 2*log_likelihood
        bic = k*np.log(n) - 2*log_likelihood
        
        return TrajectoryModel(
            model_type=model_type,
            parameters=parameters,
            r_squared=r_squared,
            aic=aic,
            bic=bic,
            rmse=rmse,
            turning_points=turning_points,
            inflection_points=inflection_points,
            monotonicity=monotonicity,
            predict_function=predict_func
        )
    
    def select_best_model(self, fitted_models: Dict[ModelType, TrajectoryModel],
                         criterion: str = "aic") -> TrajectoryModel:
        """
        Select the best model using information criteria.
        
        Args:
            fitted_models: Dictionary of fitted models
            criterion: Selection criterion ("aic", "bic", "r_squared")
            
        Returns:
            Best TrajectoryModel according to criterion
        """
        
        if not fitted_models:
            raise ValueError("No fitted models provided")
        
        if criterion == "aic":
            best_model = min(fitted_models.values(), key=lambda m: m.aic)
        elif criterion == "bic":
            best_model = min(fitted_models.values(), key=lambda m: m.bic)
        elif criterion == "r_squared":
            best_model = max(fitted_models.values(), key=lambda m: m.r_squared)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return best_model
    
    def detect_change_points(self, time_points: np.ndarray, values: np.ndarray,
                           method: ChangePointMethod = ChangePointMethod.CUSUM,
                           min_segment_length: int = 5) -> List[ChangePoint]:
        """
        Detect statistically significant change points in a time series.
        
        Mathematical Methods:
        - CUSUM: Cumulative sum control chart
        - Bayesian: Bayesian change point detection
        - Variance: Change in variance test
        
        Args:
            time_points: Time coordinates
            values: Time series values
            method: Detection method
            min_segment_length: Minimum length between change points
            
        Returns:
            List of detected ChangePoint objects
        """
        
        if len(values) < 2 * min_segment_length:
            return []
        
        if method == ChangePointMethod.CUSUM:
            return self._cusum_change_points(time_points, values, min_segment_length)
        elif method == ChangePointMethod.VARIANCE_CHANGE:
            return self._variance_change_points(time_points, values, min_segment_length)
        elif method == ChangePointMethod.BAYESIAN and STATSMODELS_AVAILABLE:
            return self._bayesian_change_points(time_points, values, min_segment_length)
        else:
            # Fallback to simple threshold method
            return self._threshold_change_points(time_points, values, min_segment_length)
    
    def _cusum_change_points(self, t: np.ndarray, y: np.ndarray, 
                           min_length: int) -> List[ChangePoint]:
        """CUSUM-based change point detection"""
        
        n = len(y)
        change_points = []
        
        # Parameters for CUSUM
        h = 4.0  # Decision threshold
        k = 0.5  # Reference value (fraction of one standard deviation)
        
        # Compute CUSUM statistics
        mean_y = np.mean(y)
        std_y = np.std(y)
        reference = k * std_y
        
        # Upper and lower CUSUM
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        
        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i-1] + (y[i] - mean_y) - reference)
            s_neg[i] = max(0, s_neg[i-1] - (y[i] - mean_y) - reference)
        
        # Detect change points
        threshold = h * std_y
        
        for i in range(min_length, n - min_length):
            if s_pos[i] > threshold or s_neg[i] > threshold:
                # Verify this is a significant change
                before_segment = y[max(0, i - min_length):i]
                after_segment = y[i:min(n, i + min_length)]
                
                if len(before_segment) > 2 and len(after_segment) > 2:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(before_segment, after_segment)
                    
                    if p_value < (1 - self.confidence_level):
                        change_point = ChangePoint(
                            location=t[i],
                            magnitude=abs(np.mean(after_segment) - np.mean(before_segment)),
                            confidence=1 - p_value,
                            change_type="mean",
                            p_value=p_value,
                            test_statistic=abs(t_stat),
                            before_mean=np.mean(before_segment),
                            after_mean=np.mean(after_segment),
                            before_variance=np.var(before_segment),
                            after_variance=np.var(after_segment)
                        )
                        change_points.append(change_point)
                        
                        # Skip ahead to avoid multiple detections of same change
                        i += min_length
        
        return change_points
    
    def _variance_change_points(self, t: np.ndarray, y: np.ndarray,
                              min_length: int) -> List[ChangePoint]:
        """Detect changes in variance using F-test"""
        
        n = len(y)
        change_points = []
        
        for i in range(min_length, n - min_length):
            before_segment = y[max(0, i - min_length):i]
            after_segment = y[i:min(n, i + min_length)]
            
            if len(before_segment) > 2 and len(after_segment) > 2:
                var_before = np.var(before_segment, ddof=1)
                var_after = np.var(after_segment, ddof=1)
                
                if var_before > 0 and var_after > 0:
                    # F-test for equality of variances
                    f_stat = var_after / var_before
                    df1 = len(after_segment) - 1
                    df2 = len(before_segment) - 1
                    
                    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2),
                                     1 - stats.f.cdf(f_stat, df1, df2))
                    
                    if p_value < (1 - self.confidence_level):
                        change_point = ChangePoint(
                            location=t[i],
                            magnitude=abs(var_after - var_before),
                            confidence=1 - p_value,
                            change_type="variance",
                            p_value=p_value,
                            test_statistic=f_stat,
                            before_mean=np.mean(before_segment),
                            after_mean=np.mean(after_segment),
                            before_variance=var_before,
                            after_variance=var_after
                        )
                        change_points.append(change_point)
        
        return change_points
    
    def _threshold_change_points(self, t: np.ndarray, y: np.ndarray,
                               min_length: int) -> List[ChangePoint]:
        """Simple threshold-based change detection (fallback method)"""
        
        # Compute moving average and detect large deviations
        window_size = min(min_length, len(y) // 4)
        if window_size < 2:
            return []
        
        moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        deviations = np.abs(np.diff(moving_avg))
        
        threshold = np.mean(deviations) + 2 * np.std(deviations)
        change_indices = np.where(deviations > threshold)[0]
        
        change_points = []
        for idx in change_indices:
            actual_idx = idx + window_size // 2
            if min_length <= actual_idx < len(t) - min_length:
                change_points.append(ChangePoint(
                    location=t[actual_idx],
                    magnitude=deviations[idx],
                    confidence=0.8,  # Rough estimate
                    change_type="trend",
                    p_value=0.05,
                    test_statistic=deviations[idx] / np.std(deviations),
                    before_mean=np.mean(y[max(0, actual_idx-min_length):actual_idx]),
                    after_mean=np.mean(y[actual_idx:min(len(y), actual_idx+min_length)]),
                    before_variance=np.var(y[max(0, actual_idx-min_length):actual_idx]),
                    after_variance=np.var(y[actual_idx:min(len(y), actual_idx+min_length)])
                ))
        
        return change_points
    
    def analyze_trajectory_complexity(self, values: np.ndarray) -> Dict[str, float]:
        """
        Analyze complexity and information content of a trajectory.
        
        Mathematical Measures:
        - Sample entropy: regularity measure
        - Fractal dimension: geometric complexity
        - Spectral entropy: frequency domain complexity
        - Approximate entropy: time series regularity
        
        Args:
            values: Trajectory values
            
        Returns:
            Dictionary of complexity measures
        """
        
        complexity_measures = {}
        
        # Sample entropy
        try:
            sample_entropy = self._compute_sample_entropy(values)
            complexity_measures['sample_entropy'] = sample_entropy
        except:
            complexity_measures['sample_entropy'] = np.nan
        
        # Spectral entropy
        try:
            spectral_entropy = self._compute_spectral_entropy(values)
            complexity_measures['spectral_entropy'] = spectral_entropy
        except:
            complexity_measures['spectral_entropy'] = np.nan
        
        # Approximate entropy
        try:
            approx_entropy = self._compute_approximate_entropy(values)
            complexity_measures['approximate_entropy'] = approx_entropy
        except:
            complexity_measures['approximate_entropy'] = np.nan
        
        # Simple variance-based complexity
        complexity_measures['variance_complexity'] = np.var(values)
        
        # Normalized trajectory length
        if len(values) > 1:
            path_length = np.sum(np.abs(np.diff(values)))
            euclidean_length = abs(values[-1] - values[0])
            complexity_measures['path_complexity'] = path_length / max(euclidean_length, 1e-6)
        else:
            complexity_measures['path_complexity'] = 0.0
        
        return complexity_measures
    
    def _compute_sample_entropy(self, values: np.ndarray, m: int = 2, r: float = None) -> float:
        """Compute sample entropy of time series"""
        
        if r is None:
            r = 0.2 * np.std(values)
        
        n = len(values)
        if n < m + 1:
            return np.nan
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi[0:m], xj[0:m])])
        
        def _phi(m):
            patterns = np.array([values[i:i + m] for i in range(n - m + 1)])
            count = np.zeros(n - m + 1)
            
            for i in range(n - m + 1):
                template = patterns[i]
                for j in range(n - m + 1):
                    if _maxdist(template, patterns[j], m) <= r:
                        count[i] += 1
            
            phi = np.sum(np.log(count / (n - m + 1))) / (n - m + 1)
            return phi
        
        return _phi(m) - _phi(m + 1)
    
    def _compute_spectral_entropy(self, values: np.ndarray) -> float:
        """Compute spectral entropy in frequency domain"""
        
        # Compute power spectral density
        freqs, psd = signal.periodogram(values)
        
        # Normalize to probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Compute entropy
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        return spectral_entropy
    
    def _compute_approximate_entropy(self, values: np.ndarray, m: int = 2, r: float = None) -> float:
        """Compute approximate entropy"""
        
        if r is None:
            r = 0.2 * np.std(values)
        
        n = len(values)
        
        def _phi(m):
            patterns = np.array([values[i:i + m] for i in range(n - m + 1)])
            count = np.zeros(n - m + 1)
            
            for i in range(n - m + 1):
                template = patterns[i]
                distances = np.max(np.abs(patterns - template), axis=1)
                count[i] = np.sum(distances <= r) / (n - m + 1)
            
            phi = np.sum(np.log(count)) / (n - m + 1)
            return phi
        
        return _phi(m) - _phi(m + 1)
    
    def compute_cross_trajectory_metrics(self, trajectories: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """
        Compute metrics describing relationships between multiple trajectories.
        
        Mathematical Analysis:
        - Cross-correlation matrix and maximum cross-correlation
        - Synchronization indices and phase coherence
        - Collective dynamics measures (order parameters)
        - Network-level statistics
        
        Args:
            trajectories: Dictionary mapping agent IDs to trajectory arrays
            
        Returns:
            Dictionary of cross-trajectory metrics
        """
        
        if len(trajectories) < 2:
            return {}
        
        # Convert to matrix format
        agent_ids = sorted(trajectories.keys())
        trajectory_matrix = np.array([trajectories[aid] for aid in agent_ids])
        n_agents, n_time = trajectory_matrix.shape
        
        metrics = {}
        
        # Cross-correlation analysis
        correlation_matrix = np.corrcoef(trajectory_matrix)
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        metrics['cross_correlation'] = {
            'mean_correlation': np.mean(upper_triangle),
            'std_correlation': np.std(upper_triangle),
            'max_correlation': np.max(upper_triangle),
            'min_correlation': np.min(upper_triangle),
            'correlation_matrix': correlation_matrix
        }
        
        # Synchronization measures
        # Kuramoto order parameter
        phases = np.angle(signal.hilbert(trajectory_matrix, axis=1))
        order_parameters = []
        
        for t in range(n_time):
            complex_order = np.mean(np.exp(1j * phases[:, t]))
            order_parameters.append(abs(complex_order))
        
        metrics['synchronization'] = {
            'mean_order_parameter': np.mean(order_parameters),
            'std_order_parameter': np.std(order_parameters),
            'max_synchronization': np.max(order_parameters),
            'order_parameter_series': order_parameters
        }
        
        # Collective dynamics
        population_mean = np.mean(trajectory_matrix, axis=0)
        population_std = np.std(trajectory_matrix, axis=0)
        
        metrics['collective_dynamics'] = {
            'population_variance': np.var(population_mean),
            'mean_individual_variance': np.mean([np.var(traj) for traj in trajectory_matrix]),
            'heterogeneity_index': np.mean(population_std),
            'consensus_trend': np.corrcoef(np.arange(n_time), population_mean)[0, 1]
        }
        
        # Clustering tendency (Hopkins statistic approximation)
        try:
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(trajectory_matrix)
            within_cluster_distances = []
            
            for i in range(n_agents):
                k_nearest = np.partition(distances[i], min(5, n_agents-1))[:min(5, n_agents-1)]
                within_cluster_distances.extend(k_nearest[k_nearest > 0])
            
            metrics['clustering'] = {
                'mean_pairwise_distance': np.mean(distances[np.triu_indices_from(distances, k=1)]),
                'clustering_tendency': np.mean(within_cluster_distances) / np.mean(distances)
            }
        except:
            metrics['clustering'] = {'clustering_tendency': np.nan}
        
        return metrics


class CrisisAnalyzer:
    """
    Specialized analyzer for crisis-driven belief dynamics.
    
    Mathematical Focus:
    - Crisis impact quantification
    - Recovery pattern analysis  
    - Intervention effectiveness measurement
    - Comparative crisis analysis
    """
    
    def __init__(self):
        self.trajectory_analyzer = TrajectoryAnalyzer()
    
    def analyze_crisis_phases(self, parameter_timeline: Dict[str, List[float]],
                            polarization_timeline: List[float]) -> Dict[str, Any]:
        """
        Identify and analyze distinct phases of crisis evolution.
        
        Mathematical Approach:
        - Phase detection using change point analysis
        - Phase characterization via parameter evolution
        - Transition dynamics quantification
        
        Args:
            parameter_timeline: Timeline of parameter evolution
            polarization_timeline: Timeline of polarization values
            
        Returns:
            Dictionary describing crisis phases
        """
        
        if not polarization_timeline:
            return {}
        
        time_points = np.arange(len(polarization_timeline))
        
        # Detect phases using change point analysis
        change_points = self.trajectory_analyzer.detect_change_points(
            time_points, np.array(polarization_timeline)
        )
        
        # Define phases based on change points
        phase_boundaries = [0] + [int(cp.location) for cp in change_points] + [len(polarization_timeline)]
        phase_boundaries = sorted(list(set(phase_boundaries)))
        
        phases = []
        for i in range(len(phase_boundaries) - 1):
            start_idx = phase_boundaries[i]
            end_idx = phase_boundaries[i + 1]
            
            phase_data = {
                'start_time': start_idx,
                'end_time': end_idx,
                'duration': end_idx - start_idx,
                'polarization_mean': np.mean(polarization_timeline[start_idx:end_idx]),
                'polarization_trend': self._compute_linear_trend(polarization_timeline[start_idx:end_idx]),
                'volatility': np.std(polarization_timeline[start_idx:end_idx])
            }
            
            # Add parameter evolution for this phase
            for param_name, values in parameter_timeline.items():
                if start_idx < len(values) and end_idx <= len(values):
                    phase_data[f'{param_name}_mean'] = np.mean(values[start_idx:end_idx])
                    phase_data[f'{param_name}_change'] = values[end_idx-1] - values[start_idx] if end_idx > start_idx else 0
            
            phases.append(phase_data)
        
        # Classify phases
        classified_phases = self._classify_crisis_phases(phases)
        
        return {
            'num_phases': len(phases),
            'phases': classified_phases,
            'change_points': [(cp.location, cp.magnitude) for cp in change_points],
            'total_duration': len(polarization_timeline),
            'crisis_intensity': max(polarization_timeline) - min(polarization_timeline)
        }
    
    def _compute_linear_trend(self, values: List[float]) -> float:
        """Compute linear trend slope"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _classify_crisis_phases(self, phases: List[Dict]) -> List[Dict]:
        """Classify crisis phases based on polarization dynamics"""
        
        classified = []
        
        for i, phase in enumerate(phases):
            phase_copy = phase.copy()
            
            # Classify based on trend and position
            if i == 0:
                phase_copy['phase_type'] = 'baseline'
            elif phase['polarization_trend'] > 0.01:
                phase_copy['phase_type'] = 'escalation'
            elif phase['polarization_trend'] < -0.01:
                phase_copy['phase_type'] = 'recovery'
            elif phase['volatility'] > np.mean([p['volatility'] for p in phases]):
                phase_copy['phase_type'] = 'turbulence'
            else:
                phase_copy['phase_type'] = 'stabilization'
            
            classified.append(phase_copy)
        
        return classified
    
    def quantify_intervention_effectiveness(self, baseline_results: List[float],
                                         intervention_results: List[float],
                                         intervention_time: int) -> Dict[str, float]:
        """
        Quantify the effectiveness of an intervention.
        
        Mathematical Measures:
        - Immediate impact: change in trajectory slope
        - Long-term effect: difference in final states
        - Recovery acceleration: time to return to baseline
        - Cost-benefit ratio: effect size vs intervention timing
        
        Args:
            baseline_results: Polarization timeline without intervention
            intervention_results: Polarization timeline with intervention
            intervention_time: Round when intervention was applied
            
        Returns:
            Dictionary of effectiveness metrics
        """
        
        if len(baseline_results) != len(intervention_results):
            warnings.warn("Timeline lengths don't match, truncating to shorter")
            min_len = min(len(baseline_results), len(intervention_results))
            baseline_results = baseline_results[:min_len]
            intervention_results = intervention_results[:min_len]
        
        if intervention_time >= len(baseline_results):
            return {'error': 'Intervention time after experiment end'}
        
        metrics = {}
        
        # Immediate impact (change in next few rounds)
        window_size = min(5, len(baseline_results) - intervention_time)
        if window_size > 0:
            pre_baseline = baseline_results[intervention_time:intervention_time + window_size]
            pre_intervention = intervention_results[intervention_time:intervention_time + window_size]
            
            immediate_impact = np.mean(pre_baseline) - np.mean(pre_intervention)
            metrics['immediate_impact'] = immediate_impact
        
        # Long-term effectiveness
        final_baseline = baseline_results[-1]
        final_intervention = intervention_results[-1]
        metrics['long_term_effect'] = final_baseline - final_intervention
        
        # Trajectory divergence over time
        divergence = np.abs(np.array(baseline_results) - np.array(intervention_results))
        metrics['max_divergence'] = np.max(divergence)
        metrics['mean_divergence'] = np.mean(divergence[intervention_time:])
        
        # Effect persistence (how long effect lasts)
        effect_threshold = 0.05
        post_intervention_divergence = divergence[intervention_time:]
        
        persistence_rounds = 0
        for div in post_intervention_divergence:
            if div > effect_threshold:
                persistence_rounds += 1
            else:
                break
        
        metrics['effect_persistence_rounds'] = persistence_rounds
        metrics['effect_persistence_ratio'] = persistence_rounds / len(post_intervention_divergence)
        
        # Cost-benefit analysis
        intervention_cost = (len(baseline_results) - intervention_time) / len(baseline_results)  # Earlier = more expensive
        benefit = abs(metrics['long_term_effect'])
        metrics['cost_benefit_ratio'] = benefit / max(intervention_cost, 0.01)
        
        # Statistical significance of effect
        if intervention_time + 5 < len(baseline_results):
            post_baseline = baseline_results[intervention_time + 5:]
            post_intervention = intervention_results[intervention_time + 5:]
            
            if len(post_baseline) > 1 and len(post_intervention) > 1:
                try:
                    t_stat, p_value = stats.ttest_ind(post_baseline, post_intervention)
                    metrics['statistical_significance'] = p_value
                    metrics['effect_size'] = abs(t_stat)
                except:
                    metrics['statistical_significance'] = np.nan
                    metrics['effect_size'] = np.nan
        
        return metrics