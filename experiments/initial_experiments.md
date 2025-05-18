
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

## Experiment 2: Symbolic–Embedding Bridge Anchors

**Objective:** Compute centroid vectors for clustered clades and classify new entities by similarity.

### Test:
- eps = 0.8
- min_samples = 2
- Added "Richard Feynman" with description: "theoretical physicist and educator"

**Result:**
- Feynman classified into `person_cluster_0` (Newton, Bohr, Einstein) with similarity 0.79
- All cities correctly grouped under `city_cluster_0`

**Conclusion:**
- Veloclade can extend symbolic knowledge through dense reasoning.
- Centroid anchors enable inference from unseen descriptions.

**Next:** Experiment 3: soft fuzzy membership assignment based on proximity to multiple centroids.

