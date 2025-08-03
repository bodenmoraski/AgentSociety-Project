"""
Enhanced Anomaly Detection for Dynamic Belief Evolution

This module provides sophisticated anomaly detection capabilities to identify
unusual patterns, outliers, and potential issues in experimental results.

Detection Categories:
1. Statistical Anomalies - Outliers in distributions
2. Temporal Anomalies - Unusual time-series patterns  
3. Behavioral Anomalies - Unexpected agent behaviors
4. Mathematical Anomalies - Violations of expected relationships
5. Crisis Response Anomalies - Unusual responses to crisis events
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class AnomalyAlert:
    """Container for anomaly detection results"""
    category: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_entities: List[Any]  # agents, rounds, parameters, etc.
    confidence: float  # 0-1
    recommendation: str
    data_points: Optional[Dict[str, Any]] = None


class EnhancedAnomalyDetector:
    """
    Sophisticated anomaly detection system for dynamic belief evolution experiments.
    
    Uses multiple detection methods:
    - Statistical outlier detection (Z-score, IQR)
    - Time series anomaly detection (change points, volatility)
    - Clustering-based anomaly detection (DBSCAN)
    - Domain-specific rule-based detection
    - Behavioral pattern analysis
    """
    
    def __init__(self, sensitivity: str = 'medium'):
        """
        Initialize detector with sensitivity level.
        
        Args:
            sensitivity: 'low', 'medium', 'high' - affects thresholds
        """
        self.sensitivity = sensitivity
        self.alerts = []
        
        # Set thresholds based on sensitivity
        if sensitivity == 'low':
            self.z_threshold = 3.0
            self.iqr_factor = 3.0
            self.change_threshold = 0.3
        elif sensitivity == 'medium':
            self.z_threshold = 2.5
            self.iqr_factor = 2.0
            self.change_threshold = 0.2
        else:  # high
            self.z_threshold = 2.0
            self.iqr_factor = 1.5
            self.change_threshold = 0.15
    
    def detect_all_anomalies(self, results) -> List[AnomalyAlert]:
        """
        Run comprehensive anomaly detection on experiment results.
        
        Args:
            results: DynamicEvolutionResults object
            
        Returns:
            List of anomaly alerts sorted by severity
        """
        
        print(f"🔍 Running Enhanced Anomaly Detection (sensitivity: {self.sensitivity})")
        print("=" * 60)
        
        self.alerts = []  # Reset alerts
        
        # 1. Statistical Anomalies
        print("📊 Detecting statistical anomalies...")
        self._detect_statistical_anomalies(results)
        
        # 2. Temporal Anomalies
        print("⏰ Detecting temporal anomalies...")
        self._detect_temporal_anomalies(results)
        
        # 3. Behavioral Anomalies
        print("🎭 Detecting behavioral anomalies...")
        self._detect_behavioral_anomalies(results)
        
        # 4. Mathematical Consistency
        print("🧮 Detecting mathematical anomalies...")
        self._detect_mathematical_anomalies(results)
        
        # 5. Crisis Response Anomalies
        print("🌊 Detecting crisis response anomalies...")
        self._detect_crisis_response_anomalies(results)
        
        # Sort by severity
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        self.alerts.sort(key=lambda x: severity_order.get(x.severity, 0), reverse=True)
        
        # Print summary
        self._print_anomaly_summary()
        
        return self.alerts
    
    def _detect_statistical_anomalies(self, results):
        """Detect statistical outliers in key metrics"""
        
        # Polarization outliers
        pol_values = np.array(results.polarization_over_time)
        z_scores = np.abs(stats.zscore(pol_values))
        outliers = np.where(z_scores > self.z_threshold)[0]
        
        if len(outliers) > 0:
            severity = 'high' if len(outliers) > len(pol_values) * 0.2 else 'medium'
            self.alerts.append(AnomalyAlert(
                category="Statistical",
                severity=severity,
                description=f"Polarization outliers detected in {len(outliers)} rounds",
                affected_entities=outliers.tolist(),
                confidence=min(1.0, np.max(z_scores[outliers]) / self.z_threshold),
                recommendation="Check for data collection errors or unexpected external events",
                data_points={'z_scores': z_scores[outliers].tolist(), 'values': pol_values[outliers].tolist()}
            ))
        
        # Agent belief distribution outliers
        if results.belief_trajectories:
            final_beliefs = [traj[-1] for traj in results.belief_trajectories.values()]
            belief_z_scores = np.abs(stats.zscore(final_beliefs))
            belief_outliers = np.where(belief_z_scores > self.z_threshold)[0]
            
            if len(belief_outliers) > 0:
                agent_ids = list(results.belief_trajectories.keys())
                outlier_agents = [agent_ids[i] for i in belief_outliers]
                
                self.alerts.append(AnomalyAlert(
                    category="Statistical",
                    severity='medium',
                    description=f"Extreme final beliefs in {len(belief_outliers)} agents",
                    affected_entities=outlier_agents,
                    confidence=min(1.0, np.max(belief_z_scores[belief_outliers]) / self.z_threshold),
                    recommendation="Investigate agent initialization or interaction patterns",
                    data_points={'final_beliefs': [final_beliefs[i] for i in belief_outliers]}
                ))
    
    def _detect_temporal_anomalies(self, results):
        """Detect unusual temporal patterns"""
        
        pol_values = np.array(results.polarization_over_time)
        
        # Sudden jumps/drops
        if len(pol_values) > 1:
            changes = np.abs(np.diff(pol_values))
            large_changes = np.where(changes > self.change_threshold)[0]
            
            if len(large_changes) > 0:
                severity = 'critical' if np.max(changes) > 0.5 else 'high'
                self.alerts.append(AnomalyAlert(
                    category="Temporal",
                    severity=severity,  
                    description=f"Sudden polarization changes in {len(large_changes)} transitions",
                    affected_entities=large_changes.tolist(),
                    confidence=min(1.0, np.max(changes[large_changes]) / self.change_threshold),
                    recommendation="Check for implementation bugs or unexpected parameter changes",
                    data_points={'changes': changes[large_changes].tolist()}
                ))
        
        # Excessive volatility
        if len(pol_values) > 3:
            volatility = np.std(np.diff(pol_values))
            if volatility > 0.1:
                self.alerts.append(AnomalyAlert(
                    category="Temporal",
                    severity='medium',
                    description=f"High polarization volatility: {volatility:.4f}",
                    affected_entities=[],
                    confidence=min(1.0, volatility / 0.1),
                    recommendation="Consider smoothing parameters or check for oscillatory behavior"
                ))
        
        # Monotonic trends (unusual)
        if len(pol_values) > 5:
            # Check for consistently increasing/decreasing trend
            diffs = np.diff(pol_values)
            consistently_increasing = np.sum(diffs > 0) / len(diffs) > 0.8
            consistently_decreasing = np.sum(diffs < 0) / len(diffs) > 0.8
            
            if consistently_increasing or consistently_decreasing:
                trend_type = "increasing" if consistently_increasing else "decreasing"
                self.alerts.append(AnomalyAlert(
                    category="Temporal",
                    severity='medium',
                    description=f"Unusually consistent {trend_type} trend",
                    affected_entities=[],
                    confidence=0.8,
                    recommendation="Verify this trend is expected given the experimental design"
                ))
    
    def _detect_behavioral_anomalies(self, results):
        """Detect unusual agent behaviors"""
        
        if not results.belief_trajectories:
            return
            
        # Agents with no belief change
        static_agents = []
        for agent_id, trajectory in results.belief_trajectories.items():
            if len(trajectory) > 1:
                total_change = abs(trajectory[-1] - trajectory[0])
                if total_change < 0.01:  # Virtually no change
                    static_agents.append(agent_id)
        
        if len(static_agents) > len(results.belief_trajectories) * 0.3:  # More than 30%
            self.alerts.append(AnomalyAlert(
                category="Behavioral",
                severity='high',
                description=f"Many agents ({len(static_agents)}) show no belief evolution",
                affected_entities=static_agents,
                confidence=0.9,
                recommendation="Check interaction mechanisms and parameter sensitivity"
            ))
        
        # Agents with extreme volatility
        volatile_agents = []
        for agent_id, trajectory in results.belief_trajectories.items():
            if len(trajectory) > 2:
                agent_volatility = np.std(np.diff(trajectory))
                if agent_volatility > 0.3:  # High volatility threshold
                    volatile_agents.append(agent_id)
        
        if len(volatile_agents) > 0:
            self.alerts.append(AnomalyAlert(
                category="Behavioral",
                severity='medium',
                description=f"Highly volatile agents detected: {len(volatile_agents)}",
                affected_entities=volatile_agents,
                confidence=0.7,
                recommendation="Investigate extreme belief oscillations in these agents"
            ))
        
        # Clustering analysis - identify isolated agents
        if len(results.belief_trajectories) > 10:
            try:
                final_beliefs = np.array([traj[-1] for traj in results.belief_trajectories.values()]).reshape(-1, 1)
                scaler = StandardScaler()
                scaled_beliefs = scaler.fit_transform(final_beliefs)
                
                # DBSCAN clustering
                clustering = DBSCAN(eps=0.5, min_samples=3).fit(scaled_beliefs)
                outliers = np.where(clustering.labels_ == -1)[0]
                
                if len(outliers) > 0:
                    agent_ids = list(results.belief_trajectories.keys())
                    outlier_agents = [agent_ids[i] for i in outliers]
                    
                    self.alerts.append(AnomalyAlert(
                        category="Behavioral",
                        severity='low',
                        description=f"Isolated agents identified through clustering: {len(outliers)}",
                        affected_entities=outlier_agents,
                        confidence=0.6,
                        recommendation="These agents may represent edge cases or minority positions"
                    ))
            except Exception:
                pass  # Skip clustering if it fails
    
    def _detect_mathematical_anomalies(self, results):
        """Detect violations of mathematical expectations"""
        
        # Check for impossible values
        pol_values = results.polarization_over_time
        impossible_pol = [i for i, p in enumerate(pol_values) if p < 0 or p > 1.2]
        
        if impossible_pol:
            self.alerts.append(AnomalyAlert(
                category="Mathematical",
                severity='critical',
                description=f"Impossible polarization values in rounds: {impossible_pol}",
                affected_entities=impossible_pol,
                confidence=1.0,
                recommendation="Critical: Fix implementation - polarization should be in [0,1] range"
            ))
        
        # Check belief trajectory bounds
        if results.belief_trajectories:
            for agent_id, trajectory in results.belief_trajectories.items():
                impossible_beliefs = [i for i, b in enumerate(trajectory) if b < -1.1 or b > 1.1]
                if impossible_beliefs:
                    self.alerts.append(AnomalyAlert(
                        category="Mathematical",
                        severity='critical',
                        description=f"Agent {agent_id} has impossible belief values",
                        affected_entities=[agent_id],
                        confidence=1.0,
                        recommendation="Critical: Fix belief bounds in agent implementation"
                    ))
                    break  # Only report first case to avoid spam
        
        # Check for conservation violations (if applicable)
        # This would depend on specific model constraints
    
    def _detect_crisis_response_anomalies(self, results):
        """Detect unusual responses to crisis scenarios"""
        
        if not hasattr(results, 'crisis_impact_metrics') or not results.crisis_impact_metrics:
            return
            
        # Check if crisis had expected impact
        polarization_increase = results.crisis_impact_metrics.get('polarization_increase', 0)
        
        if polarization_increase < 0.01:  # Very small crisis impact
            self.alerts.append(AnomalyAlert(
                category="Crisis Response",
                severity='medium',
                description=f"Crisis had minimal impact: {polarization_increase:.4f}",
                affected_entities=[],
                confidence=0.8,
                recommendation="Increase crisis severity or check crisis implementation"
            ))
        
        if polarization_increase > 0.5:  # Extremely large impact
            self.alerts.append(AnomalyAlert(
                category="Crisis Response",
                severity='high',
                description=f"Crisis had extreme impact: {polarization_increase:.4f}",
                affected_entities=[],
                confidence=0.9,
                recommendation="Verify crisis parameters are realistic"
            ))
        
        # Check recovery patterns
        recovery_ratio = results.crisis_impact_metrics.get('recovery_ratio', 0)
        if recovery_ratio < 0.1:  # No recovery
            self.alerts.append(AnomalyAlert(
                category="Crisis Response",
                severity='medium',
                description="No recovery from crisis detected",
                affected_entities=[],
                confidence=0.7,
                recommendation="Check if recovery timeline is sufficient"
            ))
    
    def _print_anomaly_summary(self):
        """Print comprehensive summary of detected anomalies"""
        
        print("\n" + "="*60)
        print("🚨 ANOMALY DETECTION SUMMARY")
        print("="*60)
        
        if not self.alerts:
            print("✅ No significant anomalies detected - system appears healthy!")
            return
        
        # Count by severity
        severity_counts = {}
        for alert in self.alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        print(f"\n📊 Total Anomalies: {len(self.alerts)}")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                icon = {'critical': '🔥', 'high': '⚠️', 'medium': '⚡', 'low': 'ℹ️'}[severity]
                print(f"   {icon} {severity.upper()}: {count}")
        
        print(f"\n🔍 Detailed Alerts:")
        for i, alert in enumerate(self.alerts[:10], 1):  # Show top 10
            severity_icon = {'critical': '🔥', 'high': '⚠️', 'medium': '⚡', 'low': 'ℹ️'}[alert.severity]
            print(f"\n{i}. {severity_icon} [{alert.category}] {alert.description}")
            print(f"   Confidence: {alert.confidence:.2f} | Affected: {len(alert.affected_entities) if alert.affected_entities else 0} entities")
            print(f"   💡 {alert.recommendation}")
        
        if len(self.alerts) > 10:
            print(f"\n... and {len(self.alerts) - 10} more alerts")
        
        # Priority recommendations
        critical_alerts = [a for a in self.alerts if a.severity == 'critical']
        if critical_alerts:
            print(f"\n🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for alert in critical_alerts:
                print(f"   • {alert.description}")
                print(f"     → {alert.recommendation}")
        
        print("\n" + "="*60)
    
    def create_anomaly_report(self, save_path: str = "anomaly_report.md"):
        """Create detailed anomaly report"""
        
        from pathlib import Path
        import time
        
        report_path = Path(save_path)
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Anomaly Detection Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Sensitivity Level:** {self.sensitivity}\n")
            f.write(f"**Total Anomalies:** {len(self.alerts)}\n\n")
            
            # Summary by category
            categories = {}
            for alert in self.alerts:
                categories[alert.category] = categories.get(alert.category, 0) + 1
            
            f.write("## Anomaly Summary by Category\n\n")
            for category, count in categories.items():
                f.write(f"- **{category}:** {count} alerts\n")
            f.write("\n")
            
            # Detailed alerts
            f.write("## Detailed Anomaly Alerts\n\n")
            for i, alert in enumerate(self.alerts, 1):
                f.write(f"### {i}. {alert.category} - {alert.severity.upper()}\n\n")
                f.write(f"**Description:** {alert.description}\n\n")
                f.write(f"**Confidence:** {alert.confidence:.2f}\n\n")
                f.write(f"**Affected Entities:** {len(alert.affected_entities) if alert.affected_entities else 0}\n\n")
                f.write(f"**Recommendation:** {alert.recommendation}\n\n")
                
                if alert.data_points:
                    f.write("**Data Points:**\n")
                    for key, value in alert.data_points.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        print(f"📋 Anomaly report saved to: {report_path}")


def run_quick_anomaly_detection():
    """Run quick anomaly detection demo"""
    
    print("🔍 Quick Anomaly Detection Demo")
    print("=" * 40)
    
    # Import here to avoid circular imports
    from experiments.dynamic_evolution.experiment import DynamicEvolutionExperiment, DynamicEvolutionConfig
    from core.dynamic_parameters import CrisisType
    from core.agent import TopicType
    
    # Create test experiment
    config = DynamicEvolutionConfig(
        name="Anomaly Detection Test",
        description="Testing anomaly detection",
        num_agents=25,
        topic=TopicType.HEALTHCARE,
        num_rounds=10,
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=0.9,  # High severity for visible effects
        interactions_per_round=40,
        belief_history_tracking=True,
        random_seed=42
    )
    
    # Run experiment
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    
    # Run anomaly detection
    detector = EnhancedAnomalyDetector(sensitivity='medium')
    alerts = detector.detect_all_anomalies(results)
    
    # Create report
    detector.create_anomaly_report("demo_outputs/anomaly_report.md")
    
    return alerts


if __name__ == "__main__":
    alerts = run_quick_anomaly_detection()