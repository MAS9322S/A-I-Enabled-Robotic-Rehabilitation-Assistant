import numpy as np
from sklearn.ensemble import RandomForestClassifier


class FormModel:
    """Educational model trained on synthetic features; not clinical data."""

    def __init__(self):
        rng = np.random.default_rng(42)
        n = 1200
        angle = rng.uniform(45, 180, n)
        angular_speed = rng.uniform(0, 220, n)
        stability = rng.uniform(0, 1, n)
        X = np.column_stack([angle, angular_speed, stability])
        quality = (
            (stability * 0.55)
            + (1 - np.minimum(np.abs(angle - 110) / 80, 1)) * 0.30
            + (1 - np.minimum(angular_speed / 220, 1)) * 0.15
        )
        y = (quality > 0.55).astype(int)
        self.model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)
        self.model.fit(X, y)

    def predict(self, angle, angular_speed, stability):
        features = np.array([[angle, angular_speed, stability]], dtype=float)
        probability = float(self.model.predict_proba(features)[0, 1])
        return probability
