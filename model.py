import numpy as np
import random
from scipy import stats

class TreeNode:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth=None, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray): X = np.array(X)
        if not isinstance(y, np.ndarray): y = np.array(y)
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * self.n_features)
        self.root = self._grow_tree(X, y, depth=0)

    def _gini(self, y):
        m = len(y)
        return 1 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes))

    def _best_split(self, X, y):
        best_gini = 1.0
        best_index, best_threshold = None, None
        features = random.sample(range(self.n_features), self.max_features)
        for idx in features:
            thresholds = np.unique(X[:, idx])
            for i in range(1, len(thresholds)):
                thr = (thresholds[i-1] + thresholds[i]) / 2
                left, right = y[X[:, idx] < thr], y[X[:, idx] >= thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = (len(left)/len(y))*self._gini(left) + (len(right)/len(y))*self._gini(right)
                if gini < best_gini:
                    best_gini = gini
                    best_index = idx
                    best_threshold = thr
        return best_index, best_threshold

    def _grow_tree(self, X, y, depth):
        class_counts = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(class_counts)
        node = TreeNode(predicted_class)
        if self.max_depth is None or depth < self.max_depth:
            index, threshold = self._best_split(X, y)
            if index is not None:
                mask = X[:, index] < threshold
                node.feature_index = index
                node.threshold = threshold
                node.left = self._grow_tree(X[mask], y[mask], depth + 1)
                node.right = self._grow_tree(X[~mask], y[~mask], depth + 1)
        return node

    # Return +1 for class=1, -1 for class=0 to use in logistic function
    def _predict_single(self, x):
        node = self.root
        while node.left:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return 1 if node.predicted_class == 1 else -1  # continuous representation

    def predict(self, X):
        if not isinstance(X, np.ndarray): X = np.array(X)
        return np.array([self._predict_single(row) for row in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, max_features=None, 
                 subsample_size=1.0, bootstrap=True, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.subsample_size = subsample_size
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def _sample(self, X, y):
        if not isinstance(X, np.ndarray): X = np.array(X)
        if not isinstance(y, np.ndarray): y = np.array(y)
        n_samples = int(len(X) * self.subsample_size)
        indices = np.random.choice(len(X), n_samples, replace=self.bootstrap)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            if self.random_state is not None:
                np.random.seed(self.random_state + i)
            tree = DecisionTree(max_depth=self.max_depth, max_features=self.max_features)
            X_sample, y_sample = self._sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # Return probabilities using RF formula
    def predict_proba(self, X):
        if not isinstance(X, np.ndarray): X = np.array(X)
        tree_outputs = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_trees, n_samples)
        probs = 1 / (1 + np.exp(-tree_outputs))  # logistic function
        avg_probs = np.mean(probs, axis=0)  # average over all trees
        return avg_probs

    # Convert probability to class label
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
