# @Time    : 10/12/2025 13:06
# @File    : DistributionGenerator.py

import numpy as np
from tqdm import tqdm
import pandas as pd
import igraph as ig
from typing import List
import re
from . import utils

"""
Cleaned DistributionGenerator with FNDD support.

FNDD supports similarity:
  - "gaussian": numeric features with RBF kernel
  - "cosine": numeric vectors with cosine similarity (mapped to [0,1])
  - "delta" / "categorical": categorical labels (recommended for MUTAG label)
"""

class DistributionGenerator:
    """
    Generator class for:
      - Node Distance Distribution (NDD)
      - Feature-weighted NDD (FNDD / fndd)
      - Transition probability matrices (tmK)
    """

    def __init__(
        self,
        distrib_type: str,
        graphs: List[ig.Graph],
        common_bin_list: bool = True,
        verbose: bool = False,
        vertex_attribute: str = None,
        feature_sigma: float = 1.0,
        similarity: str = "gaussian",
    ):
        self.verbose = verbose
        self.tqdm = tqdm if self.verbose else utils.nop
        self.distrib_type = distrib_type
        self.graph_list = graphs
        self.bin_list = None
        self.calculate_bin_list = common_bin_list
        self.distrib_list = []
        self.matcher = re.compile(r"^tm[0-9]+$")

        # FNDD options
        self.vertex_attribute = vertex_attribute
        self.feature_sigma = feature_sigma
        self.similarity = similarity

        self.__run_distib_comp()

    def get_distributions(self):
        return self.distrib_list

    def __get_bins(self):
        max_diam = max(g.diameter() for g in self.graph_list)
        self.bin_list = np.append(np.arange(0, max_diam + 1), float("inf"))

    # ----------------------------------------------------------------------
    # NDD
    # ----------------------------------------------------------------------
    def __get_node_distance_distr(self, g: ig.Graph):
        if self.bin_list is None:
            self.bin_list = np.append(np.arange(0, g.diameter() + 1), float("inf"))

        mode_g = "OUT" if g.is_directed() else "ALL"
        n = g.vcount()

        if g.is_weighted():
            d = g.shortest_paths_dijkstra(mode=mode_g, weights="weight")
        else:
            d = g.shortest_paths_dijkstra(mode=mode_g)

        h_g = np.array([np.histogram(d[i], bins=self.bin_list)[0] for i in range(n)])
        distrib_mat = h_g / max(n - 1, 1)
        self.distrib_list.append(distrib_mat)

    # ----------------------------------------------------------------------
    # FNDD
    # ----------------------------------------------------------------------
    def __get_feature_ndd(self, g: ig.Graph):
        """
        Feature-weighted Node Distance Distribution (FNDD).

        For MUTAG you should use:
          vertex_attribute="label"
          similarity="delta"  (or "categorical")

        Because MUTAG labels are categorical atom types: treating them as numeric
        with gaussian/cosine is not meaningful.
        """

        if self.bin_list is None:
            self.bin_list = np.append(np.arange(0, g.diameter() + 1), float("inf"))

        if self.vertex_attribute is None:
            raise Exception("FNDD requires vertex_attribute to be set")

        mode_g = "OUT" if g.is_directed() else "ALL"
        n = g.vcount()

        # 1) structural distances
        if g.is_weighted():
            d = np.array(g.shortest_paths_dijkstra(mode=mode_g, weights="weight"))
        else:
            d = np.array(g.shortest_paths_dijkstra(mode=mode_g))

        # 2) node attributes
        X = np.array(g.vs[self.vertex_attribute])
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 3) similarity matrix S
        sim = (self.similarity or "gaussian").lower()

        if sim == "gaussian":
            # numeric RBF kernel
            diff = X[:, None, :] - X[None, :, :]
            sqdist = np.sum(diff ** 2, axis=2)
            sigma = max(float(self.feature_sigma), 1e-8)
            sigma2 = sigma * sigma
            S = np.exp(-sqdist / (2.0 * sigma2))

        elif sim == "cosine":
            # numeric vectors cosine -> mapped to [0,1]
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            Xn = X / norms
            S = Xn @ Xn.T
            S = (S + 1.0) / 2.0

        elif sim in ("delta", "categorical"):
            # categorical: 1 if same label else 0
            if X.shape[1] == 1:
                labels = X[:, 0]
                S = (labels[:, None] == labels[None, :]).astype(float)
            else:
                # if user provides one-hot, dot product indicates match
                S = (X @ X.T)
                S = (S > 0.5).astype(float)

        else:
            raise Exception(f"Unknown similarity metric '{self.similarity}'")

        np.fill_diagonal(S, 0.0)

        # 4) weighted histogram by bins
        num_bins = len(self.bin_list) - 1
        h_g = np.zeros((n, num_bins), dtype=float)

        bin_idx = np.digitize(d, self.bin_list) - 1

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                b = bin_idx[i, j]
                if 0 <= b < num_bins:
                    h_g[i, b] += S[i, j]

        # 5) normalize per node
        row_sums = h_g.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        distrib_mat = h_g / row_sums

        self.distrib_list.append(distrib_mat)

    # ----------------------------------------------------------------------
    # TRANSITION MATRICES
    # ----------------------------------------------------------------------
    def __get_transition_matrix(self, g: ig.Graph, walk: int = 1):
        mode_g = "OUT" if g.is_directed() else "ALL"

        if walk == 1:
            if g.is_weighted():
                adj_g = g.get_adjacency(attribute="weight")
            else:
                adj_g = g.get_adjacency()

            adj = np.array(adj_g.data, dtype=float)
            dw = adj.sum(axis=0)
            dw[dw == 0] = 1.0
            distrib_mat = adj / dw

        else:
            d = g.shortest_paths_dijkstra(mode=mode_g)
            ego_out = g.neighborhood(vertices=g.vs, order=walk, mode=mode_g, mindist=walk)

            n = g.vcount()
            walk_distances = np.zeros((n, n), dtype=float)

            for i in range(n):
                for node in ego_out[i]:
                    walk_distances[i, node] = d[i][node]

            dw = walk_distances.sum(axis=0)
            dw[dw == 0] = 1.0
            distrib_mat = walk_distances / dw

        self.distrib_list.append(distrib_mat.T)

    # ----------------------------------------------------------------------
    # Dispatcher
    # ----------------------------------------------------------------------
    def __run_distib_comp(self):
        if self.distrib_type == "ndd":
            utils.vprint("Calculating Node Distance Distribution...", verbose=self.verbose)
            if self.calculate_bin_list:
                self.__get_bins()
            for g in self.tqdm(self.graph_list):
                self.__get_node_distance_distr(g)

        elif self.distrib_type == "fndd":
            utils.vprint("Calculating Feature-weighted NDD (fndd)...", verbose=self.verbose)
            if self.calculate_bin_list:
                self.__get_bins()
            for g in self.tqdm(self.graph_list):
                self.__get_feature_ndd(g)

        elif self.matcher.match(self.distrib_type):
            walk = int(self.distrib_type[2:])
            utils.vprint(f"Calculating Transition Matrices {self.distrib_type}...", verbose=self.verbose)
            for g in self.tqdm(self.graph_list):
                self.__get_transition_matrix(g, walk=walk)

        else:
            raise Exception(f"Wrong distribution selection {self.distrib_type!r}")


# ----------------------------------------------------------------------
# Standalone function (not in class)
# ----------------------------------------------------------------------
def probability_aggregator_cutoff(
    probability_distrib_matrix,
    cut_off: float = 0.01,
    agg_by: int = 5,
    return_prob: bool = True,
    remove_inf: bool = False,
):
    if remove_inf:
        probability_distrib_matrix = np.delete(
            probability_distrib_matrix,
            np.where(probability_distrib_matrix[:, -1] >= 1),
            axis=0,
        )

    if agg_by > 0:
        probability_distrib_matrix = pd.DataFrame(
            np.add.reduceat(
                probability_distrib_matrix,
                np.arange(probability_distrib_matrix.shape[1])[::agg_by],
                axis=1,
            )
        ).to_numpy()

    if cut_off > 0:
        probability_distrib_matrix[probability_distrib_matrix < cut_off] = 0

    if return_prob:
        return probability_distrib_matrix
