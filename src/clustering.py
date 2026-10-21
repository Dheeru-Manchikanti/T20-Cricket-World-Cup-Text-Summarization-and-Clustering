"""Document clustering pipeline for T20 Cricket News.

Implements TF-IDF vectorization, K-Means clustering, centroid keyword extraction,
2D TruncatedSVD projection, silhouette score evaluation, Elbow method analysis,
and out-of-sample custom document prediction.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.preprocessing import preprocess_for_vectorizer


class CricketNewsClusterer:
    """Clustering pipeline for grouping cricket news articles using TF-IDF and K-Means."""

    def __init__(
        self,
        n_clusters: int = 5,
        max_features: int = 3000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        self.requested_clusters = n_clusters
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kmeans: Optional[KMeans] = None
        self.svd: Optional[TruncatedSVD] = None
        self.feature_names: List[str] = []
        self.cluster_keywords: Dict[int, List[str]] = {}
        self.cluster_labels: Dict[int, str] = {}
        self.silhouette_avg: Optional[float] = None
        self.inertia_: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, texts: List[str]) -> "CricketNewsClusterer":
        """Fit TF-IDF vectorizer, K-Means, and 2D TruncatedSVD on the provided text documents."""
        if not texts or len(texts) < 2:
            raise ValueError("Clustering requires at least 2 non-empty articles.")

        # Dynamically bound n_clusters to usable documents count
        usable_docs = len(texts)
        self.n_clusters = max(2, min(self.requested_clusters, usable_docs - 1 if usable_docs > 2 else 2))

        # Preprocess text
        cleaned_docs = [preprocess_for_vectorizer(t) for t in texts]
        # Fallback if preprocessing cleaned too much
        cleaned_docs = [d if len(d) > 0 else "cricket tournament match" for d in cleaned_docs]

        # Fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
        )
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        if tfidf_matrix.shape[1] == 0:
            raise ValueError("Vocabulary is empty after TF-IDF vectorization.")

        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        labels = self.kmeans.fit_predict(tfidf_matrix)
        self.inertia_ = float(self.kmeans.inertia_)

        # Calculate Silhouette Score if sample size allows
        if usable_docs > self.n_clusters:
            try:
                self.silhouette_avg = float(round(silhouette_score(tfidf_matrix, labels), 3))
            except Exception:
                self.silhouette_avg = None
        else:
            self.silhouette_avg = None

        # Fit 2D TruncatedSVD for interactive visual mapping
        n_comp = min(2, tfidf_matrix.shape[1], usable_docs)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(tfidf_matrix)

        # Extract top keywords per cluster from centroids
        self._extract_cluster_metadata()
        self.is_fitted = True
        return self

    def _extract_cluster_metadata(self, top_n_terms: int = 6) -> None:
        """Derive top keywords and inferred descriptive topic labels for each cluster."""
        if self.kmeans is None or self.vectorizer is None:
            return

        self.cluster_keywords = {}
        self.cluster_labels = {}

        for c_id in range(self.n_clusters):
            centroid = self.kmeans.cluster_centers_[c_id]
            top_indices = centroid.argsort()[::-1][:top_n_terms]
            terms = [self.feature_names[i] for i in top_indices if centroid[i] > 0]

            if not terms:
                terms = [f"topic_{c_id + 1}"]

            self.cluster_keywords[c_id] = terms
            # Clear label indicating inferred cluster topic
            label_terms = ", ".join(terms[:3])
            self.cluster_labels[c_id] = f"Cluster {c_id + 1} ({label_terms})"

    def transform_2d(self, texts: List[str]) -> np.ndarray:
        """Project documents onto 2D space for interactive visualization."""
        if not self.is_fitted or self.vectorizer is None or self.svd is None:
            raise RuntimeError("Model must be fitted before transforming.")

        cleaned_docs = [preprocess_for_vectorizer(t) for t in texts]
        matrix = self.vectorizer.transform(cleaned_docs)
        coords_2d = self.svd.transform(matrix)

        # If only 1 component was fitted, pad with zeros
        if coords_2d.shape[1] < 2:
            padding = np.zeros((coords_2d.shape[0], 2 - coords_2d.shape[1]))
            coords_2d = np.hstack([coords_2d, padding])

        return coords_2d

    def get_cluster_assignments(self, texts: List[str]) -> Tuple[List[int], List[str]]:
        """Return cluster IDs and human-readable cluster topic labels for texts."""
        if not self.is_fitted or self.vectorizer is None or self.kmeans is None:
            raise RuntimeError("Model must be fitted before assigning clusters.")

        cleaned_docs = [preprocess_for_vectorizer(t) for t in texts]
        matrix = self.vectorizer.transform(cleaned_docs)
        cluster_ids = self.kmeans.predict(matrix).tolist()
        cluster_names = [self.cluster_labels.get(cid, f"Cluster {cid + 1}") for cid in cluster_ids]
        return cluster_ids, cluster_names

    def predict_custom_article(self, text: str) -> Dict[str, object]:
        """Classify a custom article into the nearest existing cluster based on cosine similarity."""
        if not self.is_fitted or self.vectorizer is None or self.kmeans is None:
            raise RuntimeError("Model must be fitted before predicting on custom text.")

        cleaned = preprocess_for_vectorizer(text)
        if not cleaned:
            cleaned = "cricket match tournament"

        vec = self.vectorizer.transform([cleaned])
        cluster_id = int(self.kmeans.predict(vec)[0])

        # Compute distance to each centroid
        distances = np.linalg.norm(self.kmeans.cluster_centers_ - vec.toarray(), axis=1)
        nearest_distance = float(round(distances[cluster_id], 4))

        keywords = self.cluster_keywords.get(cluster_id, [])
        label = self.cluster_labels.get(cluster_id, f"Cluster {cluster_id + 1}")

        # Compute top matched features for the document
        feature_scores = vec.toarray()[0]
        top_doc_indices = feature_scores.argsort()[::-1][:5]
        doc_keywords = [
            self.feature_names[i] for i in top_doc_indices if feature_scores[i] > 0
        ]

        return {
            "predicted_cluster_id": cluster_id,
            "cluster_label": label,
            "cluster_keywords": keywords,
            "document_top_keywords": doc_keywords,
            "centroid_distance": nearest_distance,
            "note": (
                "Cluster assignment is based on TF-IDF cosine/Euclidean distance to the nearest "
                "unsupervised K-Means centroid. It represents automated text similarity, "
                "not human-curated editorial classification."
            ),
        }

    def get_representative_articles(
        self, df: pd.DataFrame, top_per_cluster: int = 2
    ) -> Dict[int, pd.DataFrame]:
        """Find the articles closest to each cluster center (most prototypical articles)."""
        if not self.is_fitted or self.vectorizer is None or self.kmeans is None:
            return {}

        cleaned_docs = [preprocess_for_vectorizer(t) for t in df["article_text"]]
        matrix = self.vectorizer.transform(cleaned_docs).toarray()

        representatives: Dict[int, pd.DataFrame] = {}
        for c_id in range(self.n_clusters):
            centroid = self.kmeans.cluster_centers_[c_id]
            # Articles belonging to this cluster
            mask = df["cluster_id"] == c_id
            cluster_indices = df.index[mask].tolist()

            if not cluster_indices:
                continue

            sub_matrix = matrix[cluster_indices]
            dists = np.linalg.norm(sub_matrix - centroid, axis=1)
            closest_local_indices = dists.argsort()[:top_per_cluster]
            closest_global_indices = [cluster_indices[i] for i in closest_local_indices]

            representatives[c_id] = df.loc[closest_global_indices]

        return representatives


def compute_elbow_curve(
    texts: List[str], max_k: int = 8, random_state: int = 42
) -> Dict[str, Union[List[int], List[float]]]:
    """Compute Inertia and Silhouette scores across multiple k values for Elbow analysis."""
    if not texts or len(texts) < 3:
        return {"k_values": [], "inertias": [], "silhouettes": []}

    cleaned_docs = [preprocess_for_vectorizer(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=2500, stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(cleaned_docs)

    max_possible_k = min(max_k, len(texts) - 1)
    k_range = list(range(2, max(3, max_possible_k + 1)))

    inertias: List[float] = []
    silhouettes: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
        labels = km.fit_predict(matrix)
        inertias.append(float(round(km.inertia_, 2)))

        try:
            sil = float(round(silhouette_score(matrix, labels), 3))
        except Exception:
            sil = 0.0
        silhouettes.append(sil)

    return {"k_values": k_range, "inertias": inertias, "silhouettes": silhouettes}
