import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler


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
    def extract_features(self, parcel, system) -> np.ndarray:
        # ── static parcel features ───────────────────────────
        vol    = parcel.get_volume()
        wt     = parcel.weight
        feas   = list(parcel.feasible_outfeeds)
        m      = len(feas)
        min_idx = float(min(feas)) if m else -1.0
        avg_idx = float(sum(feas)/m) if m else -1.0

        # ── dynamic loads for those feasible gates ──────────
        if m:
            time_loads = np.array([system.loads[k]   for k in feas])
            len_loads  = np.array([system.loads_l[k] for k in feas])
            rem_caps   = np.array([
                system.outfeeds[k].max_length - system.loads_l[k]
                for k in feas
            ])
            dists      = np.array([
                system.dist_scanner_to_outfeeds + k * system.dist_between_outfeeds
                for k in feas
            ])
        else:
            # no feas gates → dummy arrays so stats() won’t crash
            time_loads = len_loads = rem_caps = dists = np.array([0.0])

        # helper: min/mean/max/std
        def stats(arr):
            return float(arr.min()), float(arr.mean()), float(arr.max()), float(arr.std())

        t_min, t_avg, t_max, t_std = stats(time_loads)
        l_min, l_avg, l_max, l_std = stats(len_loads)
        r_min, r_avg, r_max, r_std = stats(rem_caps)
        d_min, d_avg, d_max, d_std = stats(dists)

        # global imbalance (time-based)
        all_time = np.array(list(system.loads.values()))
        imb_time = float(all_time.max() - all_time.min())

        # recirculation count
        recirc = float(parcel.recirculation_count)

        # combine into one vector
        feats = [
            vol, wt, m, min_idx, avg_idx,
            t_min, t_avg, t_max, t_std,
            l_min, l_avg, l_max, l_std,
            r_min, r_avg, r_max, r_std,
            d_min, d_avg, d_max, d_std,
            recirc,
            imb_time,
        ]
        return np.array(feats, dtype=float)