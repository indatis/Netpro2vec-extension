import igraph as ig
from netpro2vec.Netpro2vec import Netpro2vec

# ----------------------------------------------
# Minimal graph for debugging
# ----------------------------------------------
g = ig.Graph()
g.add_vertices(4)
g.add_edges([(0,1), (1,2), (2,3)])

# Add a simple node feature for fndd
g.vs["feat"] = [0.1, 0.2, 0.25, 0.3]

graphs = [g]

# ----------------------------------------------
# Test 1: original NDD
# ----------------------------------------------
print("Running NDD...")
model_ndd = Netpro2vec(
    prob_type=["ndd"],
    extractor=[1],
    agg_by=[1],
    vertex_attribute=None,
    verbose=True
)

try:
    model_ndd.fit(graphs)
    print("NDD OK.")
except Exception as e:
    print("NDD ERROR:", e)


# ----------------------------------------------
# Test 2: new FNDD
# ----------------------------------------------
print("\nRunning FNDD...")
model_fndd = Netpro2vec(
    prob_type=["fndd"],
    extractor=[1],
    agg_by=[1],
    vertex_attribute="feat",
    feature_sigma=0.1,
    similarity="gaussian",
    verbose=True
)

try:
    model_fndd.fit(graphs)
    print("FNDD OK.")
except Exception as e:
    print("FNDD ERROR:", e)
