# ml_sorting.py

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

class OutfeedML:
    """
    A simple wrapper around a RandomForestClassifier (or any sklearn classifier)
    to predict the “best” outfeed ID for a given parcel, based on its features.
    
    USAGE:
      1. Instantiate: model = OutfeedML()
      2. Train: 
         - Prepare X_train: a 2D array of shape (n_samples, n_features)
         - Prepare y_train: a 1D array of shape (n_samples,) containing outfeed IDs
         - Call model.fit(X_train, y_train)
         - Optionally: model.save("my_model.pkl")
      3. Inference:
         - Call model.predict(parcel) → returns a single integer (outfeed_id)
         - Or if you want probabilities: model.predict_proba(parcel)
      4. To load a pretrained model: model = OutfeedML(model_path="my_model.pkl")
    """

    def __init__(self, model_path: str = None, n_estimators: int = 100, random_state: int = 42):
        """
        If model_path is given, we load from disk; otherwise, we create a fresh RandomForestClassifier.
        """
        if model_path:
            self.clf = joblib.load(model_path)
            self.is_trained = True
        else:
            self.clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
            self.is_trained = False

    @staticmethod
    def parcel_to_features(parcel) -> np.ndarray:
        """
        Extracts a 1D feature vector from a Parcel instance. 
        You can add or remove features here as needed.
        
        Currently using:
          - volume = length * width * height
          - weight
          - number of feasible outfeeds
        """
        volume = parcel.length * parcel.width * parcel.height
        weight = parcel.weight
        num_choices = len(parcel.feasible_outfeeds)
        # You could also one-hot encode each feasible_outfeed, etc.
        return np.array([volume, weight, num_choices], dtype=float)

    def fit(self, parcels: list, labels: list[int]) -> None:
        """
        Train the classifier on a list of “parcel” objects and their corresponding outfeed IDs.
        
        parcels:   [Parcel, Parcel, …]
        labels:    [int,    int,    …]  (the “true” outfeed ID for each parcel)
        """
        if len(parcels) != len(labels):
            raise ValueError("Number of parcels and labels must match.")
        
        X = np.vstack([self.parcel_to_features(p) for p in parcels])
        y = np.array(labels, dtype=int)
        self.clf.fit(X, y)
        self.is_trained = True

    def predict(self, parcel) -> int:
        """
        Given a single Parcel instance, return a single outfeed ID (int).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call .fit(...) first.")
        
        feats = self.parcel_to_features(parcel).reshape(1, -1)  # shape = (1, n_features)
        pred = self.clf.predict(feats)  # returns array([outfeed_id])
        return int(pred[0])

    def predict_proba(self, parcel) -> np.ndarray:
        """
        Return the probability distribution over all outfeed IDs.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call .fit(...) first.")
        
        feats = self.parcel_to_features(parcel).reshape(1, -1)
        return self.clf.predict_proba(feats)[0]  # e.g. array([0.1, 0.7, 0.2]) if 3 classes

    def save(self, model_path: str) -> None:
        """Save the trained model to disk using joblib."""
        if not self.is_trained:
            raise RuntimeError("Nothing to save; model is not trained.")
        joblib.dump(self.clf, model_path)

    @classmethod
    def load(cls, model_path: str):
        """Convenience constructor to load a pretrained model from disk."""
        return cls(model_path=model_path)
