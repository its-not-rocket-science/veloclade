
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

def cluster_and_expand(parent_class, eps=0.5, min_samples=2):
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

def example_veloclade():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    entity = Entity("entity")
    organism = Entity("organism")
    person = Entity("person")
    location = Entity("location")
    city = Entity("city")
    entity.add_subclass(organism)
    entity.add_subclass(location)
    organism.add_subclass(person)
    location.add_subclass(city)
    dataset = [
        ("Albert Einstein", "famous theoretical physicist"),
        ("Marie Curie", "pioneer in radioactivity research"),
        ("Isaac Newton", "mathematician and physicist"),
        ("Paris", "capital city of France"),
        ("London", "capital city of UK"),
        ("Tokyo", "capital city of Japan")
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
    cluster_and_expand(person, eps=0.7, min_samples=2)
    cluster_and_expand(city, eps=0.7, min_samples=2)
    print("\nVeloclade Structure:")
    for subclass in person.subclasses:
        print(f"{person.name} → {subclass.name}: {[e.name for e in subclass.instances]}")
    for subclass in city.subclasses:
        print(f"{city.name} → {subclass.name}: {[e.name for e in subclass.instances]}")

if __name__ == "__main__":
    example_veloclade()
