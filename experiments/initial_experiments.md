
# Veloclade - Initial Experiments

## Experiment 1: Semantic Clustering Sensitivity

**Objective:** Test DBSCAN clustering of sentence-transformer embeddings to drive dynamic subclassing.

### Test 1.1 - Varying Descriptions
- Dataset: 6 scientists, 6 cities
- Result: Only `person_cluster_0` was created (Bohr, Einstein, Newton)
- No city subclassing due to diverse phrasing of descriptions

### Test 1.2 - Standardized City Descriptions
- Changed all city descriptions to: "major capital city"
- Result: City embeddings aligned; DBSCAN created 1 or more city subclasses

**Conclusion:** 
- Semantic normalization is critical for embedding-based clustering.
- Sentence phrasing affects vector similarity and symbolic structure formation.

**Next:** Use clustering results to define centroid embeddings for each subclass, enabling symbolic–dense bridges.
