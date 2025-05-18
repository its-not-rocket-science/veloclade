
# Experiment 2 Notes

**Goal:** Link symbolic subclasses to their embedding-space centroids and use these to classify new entities by similarity.

**Setup:**
- Clustered known entities using DBSCAN.
- Computed the centroid embedding for each resulting subclass.
- Embedded a new entity ("Richard Feynman") and compared it to the subclass centroids.
- Suggested a clade based on cosine similarity.

**Dataset:**
- 6 scientists
- 6 cities (standardized description: "major capital city")

**Results:**
- `person_cluster_0`: ['Isaac Newton', 'Niels Bohr', 'Albert Einstein']
- `city_cluster_0`: ['New York', 'Berlin', 'London', 'Tokyo', 'Paris', 'Rome']
- Feynman matched `person_cluster_0` with similarity 0.79

**Findings:**
- Symbolic clades can serve as semantic anchors in dense vector space.
- Sentence embeddings allow meaningful, concept-level classification of new entities.
- Semantic normalization of descriptions remains essential for clustering success.

**Next Steps:**
- Extend similarity matching to support soft/fuzzy membership across multiple clades.
- Log and assign confidence-weighted relationships instead of hard assignments.
