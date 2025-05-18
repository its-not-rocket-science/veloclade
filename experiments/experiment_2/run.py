from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np


class Entity:
    def __init__(self, name, description=None):
        self.name = name
        self.description = description or name
        self.subclasses = set()
        self.superclasses = set()
        self.instances = set()
        self.properties = dict()
        self.fuzzy_memberships = dict()
        self.embedding = None
        self.centroid = None  # for subclasses only

    def add_subclass(self, subclass):
        self.subclasses.add(subclass)
        subclass.superclasses.add(self)

    def add_instance(self, instance, confidence=1.0):
        self.instances.add(instance)
        instance.fuzzy_memberships[self] = confidence
        instance.superclasses.add(self)

    def add_property(self, key, value):
        self.properties[key] = value

    def get_instances(self):
        return self.instances

    def __repr__(self):
        return f"Entity({self.name})"


def encode_entities(entities, model):
    sentences = [e.description for e in entities]
    embeddings = model.encode(sentences)
    for e, emb in zip(entities, embeddings):
        e.embedding = emb


def compute_centroid(entities):
    vectors = [e.embedding for e in entities]
    return np.mean(vectors, axis=0)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cluster_and_expand(parent_class, eps=0.6, min_samples=2):
    instances = list(parent_class.get_instances())
    if len(instances) < min_samples:
        return
    X = np.array([e.embedding for e in instances])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    cluster_map = {}
    for label, entity in zip(labels, instances):
        if label == -1:
            continue
        if label not in cluster_map:
            new_subclass = Entity(f"{parent_class.name}_cluster_{label}")
            parent_class.add_subclass(new_subclass)
            cluster_map[label] = new_subclass
        parent_class.instances.remove(entity)
        cluster_map[label].add_instance(entity)

    # Compute and assign centroid embeddings
    for label, subclass in cluster_map.items():
        subclass.centroid = compute_centroid(list(subclass.instances))


def run_experiment_2():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Base classes
    entity = Entity("entity")
    organism = Entity("organism")
    person = Entity("person")
    location = Entity("location")
    city = Entity("city")
    entity.add_subclass(organism)
    entity.add_subclass(location)
    organism.add_subclass(person)
    location.add_subclass(city)

    # Dataset with standardized city descriptions
    dataset = [
        ("Albert Einstein", "famous theoretical physicist"),
        ("Marie Curie", "pioneer in radioactivity research"),
        ("Isaac Newton", "mathematician and physicist"),
        ("Niels Bohr", "quantum theory pioneer"),
        ("Stephen Hawking", "black hole and relativity physicist"),
        ("Ada Lovelace", "early computing visionary"),
        ("Paris", "major capital city"),
        ("London", "major capital city"),
        ("Tokyo", "major capital city"),
        ("New York", "major capital city"),
        ("Rome", "major capital city"),
        ("Berlin", "major capital city")
    ]

    people = []
    places = []
    for name, desc in dataset:
        ent = Entity(name, desc)
        if "city" in desc:
            city.add_instance(ent)
            places.append(ent)
        else:
            person.add_instance(ent)
            people.append(ent)

    encode_entities(people + places, model)

    cluster_and_expand(person, eps=0.8, min_samples=2)
    cluster_and_expand(city, eps=0.8, min_samples=2)

    print("\nExperiment 2: Symbolic–Embedding Bridge Anchors")
    for subclass in person.subclasses:
        members = [e.name for e in subclass.instances]
        print(f"{person.name} → {subclass.name}: {members}")
    for subclass in city.subclasses:
        members = [e.name for e in subclass.instances]
        print(f"{city.name} → {subclass.name}: {members}")

    # Test entity classification by centroid similarity
    print("\nTesting new entity classification:")
    new_entity = Entity("Richard Feynman",
                        "theoretical physicist and educator")
    encode_entities([new_entity], model)

    best_match = None
    best_score = -1

    for subclass in person.subclasses:
        if subclass.centroid is not None:
            sim = cosine_similarity(new_entity.embedding, subclass.centroid)
            print(f"Similarity to {subclass.name}: {sim:.2f}")
            if sim > best_score:
                best_score = sim
                best_match = subclass

    if best_match:
        print(
            f"→ Suggested clade for '{new_entity.name}': {best_match.name} (score: {best_score:.2f})")
    else:
        print(f"No suitable subclass found for '{new_entity.name}'.")


if __name__ == "__main__":
    run_experiment_2()
