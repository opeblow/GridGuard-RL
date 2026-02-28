import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import joblib


class AnomalyDetector:
    """Detecting grid instability using anomaly detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        self.is_fitted = False
        self.threshold = 0.5
        self.score_min = None
        self.score_max = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fitting the anomaly detector on training data"""
        X_scaled = self.scaler.fit_transform(X)
        
        if y is not None:
            self.classifier.fit(X_scaled, y)
            self.iso_forest.fit(X_scaled)
        else:
            self.iso_forest.fit(X_scaled)
        
        train_scores = self.iso_forest.decision_function(X_scaled)
        self.score_min = train_scores.min()
        self.score_max = train_scores.max()
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicting anomaly scores (-1 for anomaly, 1 for normal)"""
        if not self.is_fitted:
            raise ValueError("Detector not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.classifier, 'classes_'):
            return self.classifier.predict(X_scaled)
        return self.iso_forest.predict(X_scaled)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Getting anomaly scores (lower = more anomalous)"""
        if not self.is_fitted:
            raise ValueError("Detector not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.classifier, 'classes_'):
            proba = self.classifier.predict_proba(X_scaled)
            return proba[:, 1]
        return self.iso_forest.decision_function(X_scaled)
    
    def get_instability_score(self, state: np.ndarray) -> float:
        """Getting normalized instability score [0, 1]"""
        if not self.is_fitted:
            return 0.0
            
        score = self.score(state.reshape(1, -1))[0]
        
        if hasattr(self.classifier, 'classes_'):
            return float(np.clip(score, 0.0, 1.0))
            
        if self.score_max == self.score_min:
            return 0.5
            
        normalized = 1.0 - (score - self.score_min) / (self.score_max - self.score_min)
        return np.clip(normalized, 0.0, 1.0)
    
    def save(self, path: str):
        """Saving the detector"""
        joblib.dump({
            'scaler': self.scaler,
            'iso_forest': self.iso_forest,
            'threshold': self.threshold,
            'score_min': self.score_min,
            'score_max': self.score_max
        }, path)
        
    def load(self, path: str):
        """Loading the detector"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.iso_forest = data['iso_forest']
        self.threshold = data['threshold']
        self.score_min = data.get('score_min', None)
        self.score_max = data.get('score_max', None)
        self.is_fitted = True


class InstabilityPredictor:
    """Predicting future instability based on current trends"""
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.state_history = []
        self.thresholds = {
            'voltage': 0.9,
            'loading': 0.85,
            'instability': 0.6
        }
        
    def update(self, state: np.ndarray) -> Dict:
        """Updating with new state and get warning"""
        self.state_history.append(state)
        
        if len(self.state_history) > self.lookback:
            self.state_history.pop(0)
            
        warning = self._analyze_trends()
        return warning
    
    def _analyze_trends(self) -> Dict:
        """Analyzing trends in state history"""
        if len(self.state_history) < 3:
            return {'level': 'normal', 'confidence': 0.0}
            
        states = np.array(self.state_history)
        
        voltages = states[:, :14]
        loadings = states[:, 14:30]
        
        voltage_trend = np.mean(voltages[-1] - voltages[0])
        loading_trend = np.mean(loadings[-1] - loadings[0])
        
        voltage_min = np.min(voltages[-1])
        loading_max = np.max(loadings[-1])
        
        level = 'normal'
        confidence = 0.0
        recommended_actions = []
        
        if voltage_min < self.thresholds['voltage'] * 0.8:
            level = 'critical'
            confidence = 0.9
            recommended_actions = ['emergency_load_shedding', 'island_critical_loads']
        elif voltage_min < self.thresholds['voltage']:
            level = 'warning'
            confidence = 0.7
            recommended_actions = ['redispatch_generation', 'reduce_load']
        elif loading_max > self.thresholds['loading']:
            level = 'warning'
            confidence = 0.6
            recommended_actions = ['redispatch_generation', 'activate_reserves']
            
        if voltage_trend < -0.01:
            confidence = min(confidence + 0.2, 1.0)
            
        if loading_trend > 0.05:
            confidence = min(confidence + 0.2, 1.0)
            
        return {
            'level': level,
            'confidence': confidence,
            'voltage_min': voltage_min,
            'loading_max': loading_max,
            'voltage_trend': voltage_trend,
            'loading_trend': loading_trend,
            'recommended_actions': recommended_actions
        }
    
    def reset(self):
        """Resetting history"""
        self.state_history = []


class EarlyWarningSystem:
    """Early warning system for grid instability"""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.predictor = InstabilityPredictor()
        self.warning_threshold = 0.6
        self.alert_history = []
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Training the warning system"""
        self.anomaly_detector.fit(X, y)
        
    def assess(self, state: np.ndarray) -> Dict:
        """Assessing current grid state"""
        instability = self.anomaly_detector.get_instability_score(state)
        
        warning = self.predictor.update(state)
        warning['instability_score'] = instability
        
        if instability > self.warning_threshold:
            warning['alert'] = 'ELEVATED'
        if instability > 0.8:
            warning['alert'] = 'CRITICAL'
            
        self.alert_history.append(warning)
        
        return warning
    
    def get_lead_time_estimate(self) -> int:
        """Estimating minutes to potential failure"""
        if len(self.alert_history) < 5:
            return -1
            
        recent = self.alert_history[-5:]
        
        if all(a['instability_score'] > 0.7 for a in recent):
            return 15
        elif all(a['instability_score'] > 0.5 for a in recent):
            return 30
        elif any(a['instability_score'] > 0.6 for a in recent):
            return 45
            
        return -1
    
    def save(self, path: str):
        """Saving the system"""
        self.anomaly_detector.save(path + '_anomaly.pkl')
        
    def load(self, path: str):
        """Load the system"""
        self.anomaly_detector.load(path + '_anomaly.pkl')
