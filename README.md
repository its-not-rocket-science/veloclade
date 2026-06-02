# Veloclade

A research prototype of a neuro-symbolic knowledge graph system with clade-inspired ontology growth control.

Veloclade uses a hierarchical classification structure inspired by biological cladistics — combined with sentence-transformer embedding clustering — to control the growth of ontologies and mitigate *subclassing explosion*: the tendency of knowledge graphs to proliferate redundant, overly specific subclasses that fragment reasoning and inflate graph size.

---

## The problem

In large ontologies, subclassing explosion is pervasive. Every time a new concept is added, the path of least resistance is to create a new subclass of something nearby. The result is ontologies thousands of nodes deep with near-duplicate classes, poor generalisation, and brittle inference. Existing approaches (OWL reasoners, manual curation) do not scale.

Veloclade approaches this differently: rather than preventing new nodes, it controls *where they go* in the hierarchy by combining:

1. **Clade-based hierarchy** — inheritance is governed by clade membership (monophyletic groupings), not arbitrary parent–child relationships. New concepts must justify their position in terms of shared derived properties, not just surface similarity.
2. **Embedding clustering** — sentence-transformer embeddings of concept descriptions are clustered to identify when a proposed new node is semantically redundant with an existing one, or belongs within an existing clade rather than spawning a new one.
3. **Growth policy engine** — configurable rules governing when a new node may be added, when it should be merged with an existing node, and when a new clade may be created.

---

## Architecture

```
New concept proposal
        │
        ▼
┌───────────────────────┐
│  Embedding encoder     │  — sentence-transformers
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Clade membership      │  — Find nearest existing clade
│  classifier            │    by embedding similarity
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Growth policy engine  │  — Merge / place in clade / create new clade
│                        │    based on configurable thresholds
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Knowledge graph       │  — RDFLib / custom graph store
│  update                │
└───────────────────────┘
```

---

## Getting started

```bash
pip install -r requirements.txt
python -m veloclade.demo
```

This runs a demonstration of controlled ontology growth on a sample biological classification dataset, showing subclassing explosion in an unconstrained graph versus controlled growth under Veloclade's policy engine.

### Requirements

- Python 3.10+
- sentence-transformers
- RDFLib
- scikit-learn

---

## Related projects

Veloclade is part of a cluster of related knowledge representation tools:

- [koios](../koios) — ontology-grounded transformer for knowledge-augmented reasoning
- [ananke](../ananke) — ontology-driven world-building system (a practical application of controlled ontology growth)

---

## Related work

- Kulmanov et al. (2019) — [ELEmbeddings: Geometric construction of models for the Description Logic EL++](https://arxiv.org/abs/1902.10499)
- Chen et al. (2021) — [Ontology-Enhanced Pre-training for Bio-Medical NLP](https://arxiv.org/abs/2110.05572)

---

## Status

Research prototype. Core clade membership classifier and growth policy engine are implemented. Evaluation against standard ontology benchmarks (OWL ontologies from BioPortal) is planned.

---

## Licence

MIT
