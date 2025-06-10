import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    RandomOverSampler = None

class OutfeedML:
    """
    A RandomForest-based gate selector that:
      1) augments the feature vector with gate-index stats,
      2) oversamples late-gate classes at train time (if imblearn present),
      3) exposes a confidence threshold for runtime fallback.
    """

    def __init__(
        self,
        model_path: str = None,
        n_estimators: int = 100,
        random_state: int = 42,
        threshold: float = 0.5
    ):
        if model_path:
            self.clf = joblib.load(model_path)
            self.is_trained = True
        else:
            self.clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight='balanced'
            )
            self.is_trained = False

        self.random_state = random_state
        self.threshold = threshold

    @staticmethod
    def parcel_to_features(parcel) -> np.ndarray:
        """
        Build a 5-dim feature vector:
          [ volume, weight, num_choices, min_outfeed_idx, avg_outfeed_idx ]
        """
        volume = parcel.length * parcel.width * parcel.height
        weight = parcel.weight

        feas = parcel.feasible_outfeeds
        m = len(feas)
        if m > 0:
            min_idx = float(min(feas))
            avg_idx = float(sum(feas) / m)
        else:
            # no mechanically feasible gate
            min_idx = -1.0
            avg_idx = -1.0

        return np.array([volume, weight, m, min_idx, avg_idx], dtype=float)

    def fit(self, parcels: list, labels: list[int]) -> None:
        """
        Train on (parcels, labels). Optionally oversample late-gate classes.
        """
        if len(parcels) != len(labels):
            raise ValueError("Number of parcels and labels must match.")

        # Build feature matrix
        X = np.vstack([self.parcel_to_features(p) for p in parcels])
        y = np.array(labels, dtype=int)

        # Oversample under-represented classes if imblearn is available
        if RandomOverSampler is not None:
            ros = RandomOverSampler(random_state=self.random_state)
            X, y = ros.fit_resample(X, y)

        # Fit the forest
        self.clf.fit(X, y)
        self.is_trained = True

    def predict(self, parcel) -> int:
        """
        Predict the single best gate index.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet; call .fit(...) first.")

        feats = self.parcel_to_features(parcel).reshape(1, -1)
        return int(self.clf.predict(feats)[0])

    def predict_proba(self, parcel) -> np.ndarray:
        """
        Return class-probabilities over all gates.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet; call .fit(...) first.")

        feats = self.parcel_to_features(parcel).reshape(1, -1)
        return self.clf.predict_proba(feats)[0]

    def save(self, model_path: str) -> None:
        """
        Serialize the trained forest.
        """
        if not self.is_trained:
            raise RuntimeError("Nothing to save; model is not trained.")
        joblib.dump(self.clf, model_path)

    @classmethod
    def load(cls, model_path: str, **kwargs):
        """
        Convenience loader: returns an instance with .is_trained = True.
        """
        inst = cls(model_path=None, **kwargs)
        inst.clf = joblib.load(model_path)
        inst.is_trained = True
        return inst
