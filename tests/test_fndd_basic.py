# tests/test_fndd_basic.py

import igraph as ig
import numpy as np
from netpro2vec.Netpro2vec import Netpro2vec

print("\nRunning NDD...")

# ------------------------------------------------------------
# 1. Build a nontrivial graph so histograms are NOT empty
# ------------------------------------------------------------
g = ig.Graph.Erdos_Renyi(n=20, p=0.25)
g.vs["feat"] = np.random.rand(20)   # needed for FNDD later

graphs = [g]

# ------------------------------------------------------------
# 2. Test Standard NDD
# ------------------------------------------------------------
try:
    model_ndd = Netpro2vec(
        prob_type=["ndd"],
        extractor=[1],
        agg_by=[1],
        verbose=True
    )
    model_ndd.fit(graphs)
    emb_ndd = model_ndd.get_embedding()
    print("NDD OK. Embedding shape:", emb_ndd.shape)
except Exception as e:
    print("NDD ERROR:", e)


print("\nRunning FNDD...")

# ------------------------------------------------------------
# 3. Test Feature-weighted NDD
# ------------------------------------------------------------
try:
    model_fndd = Netpro2vec(
        prob_type=["fndd"],
        extractor=[1],
        agg_by=[1],
        vertex_attribute="feat",
        feature_sigma=1.0,
        similarity="gaussian",
        verbose=True
    )
    model_fndd.fit(graphs)
    emb_fndd = model_fndd.get_embedding()
    print("FNDD OK. Embedding shape:", emb_fndd.shape)
except Exception as e:
    print("FNDD ERROR:", e)
