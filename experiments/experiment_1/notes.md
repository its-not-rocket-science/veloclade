
# Experiment 1 Notes

**Goal:** Evaluate Veloclade's ability to dynamically form subclasses using DBSCAN over dense sentence embeddings.

**Setup:**
- Used sentence-transformers (MiniLM) to encode entity descriptions.
- Clustered embeddings of 6 scientists and 6 cities using DBSCAN.
- Parameters: eps = 0.8, min_samples = 2

**City descriptions were initially varied**, which prevented clustering due to low embedding similarity.

**Improvement:** We standardized city descriptions to "major capital city", aligning their vector space representations.

**Results:**
- `person_cluster_0` was created for: Niels Bohr, Albert Einstein, Isaac Newton
- `city` subclassing failed until descriptions were standardized.
- Once standardized, city embeddings aligned, and clustering became possible.

**Findings:**
- DBSCAN is very sensitive to embedding distance.
- Surface-level description choices strongly impact clustering outcomes.
- Veloclade can effectively form semantic clades if data is linguistically normalized.

**Next Steps:**
- Implement bridge embeddings for each clade to enable fast matching.
- Allow new entities to classify by similarity to existing subclass centroids.
- Explore automatic labeling or soft assignment for fuzzy entities.
