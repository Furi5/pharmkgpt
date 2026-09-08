# PharmKGPT local release adaptation, 2026-09-08. See kg_construction/SOURCE_NOTES.md.
from pydantic import BaseModel, SkipValidation
from typing import Callable
import numpy as np
import re
import logging

class EntityProperties(BaseModel):
    embeddings: SkipValidation[np.array]=None
    class Config:
        arbitrary_types_allowed = True

class RelationshipProperties(BaseModel):
    embeddings:SkipValidation[np.array]=None
    class Config:
        arbitrary_types_allowed = True

class Entity(BaseModel):
    label:str = ""
    name:str = ""
    properties:EntityProperties = EntityProperties()
    properties_info:dict = {}

    def process(self):
        # Replace spaces, dashes, periods, and '&' in names with underscores or 'and'.
        self.label = re.sub(r'[^a-zA-Z0-9]', '_', self.label).replace("&", "and")
        self.name = self.name.lower().replace("_", " ").replace("-", " ").replace('"', " ").strip()

    def embed_Entity(self,
                     embeddings_function:Callable[[str], np.array],
                     entity_name_weight:float=0.6,
                     entity_label_weight:float=0.4)-> None:
        self.process()
        self.properties.embeddings = (
            entity_name_weight * embeddings_function(self.name)
            +
            entity_label_weight * embeddings_function(self.label)
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, Entity):
            return self.name == other.name and self.label == other.label
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.label))

    def __repr__(self):
        return f"Entity(name={self.name}, label={self.label}, properties={self.properties}, properties_info={self.properties_info})"

class Relationship(BaseModel):
    startEntity:Entity = Entity()
    endEntity:Entity = Entity()
    name:str = ""
    properties:RelationshipProperties = RelationshipProperties()
    properties_info:dict = {}

    def process(self):
        # Replace spaces, dashes, periods, and '&' in names with underscores or 'and'.
        self.name = re.sub(r'[^a-zA-Z0-9]', '_', self.name).replace("&", "and")

    def embed_relationship(self, embeddings_function:Callable[[str], np.array]):
        self.process()
        self.properties.embeddings = embeddings_function(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, Relationship):
            return (self.startEntity == other.startEntity
                    and self.endEntity == other.endEntity
                    and self.name == other.name
                    )
        return False

    def __hash__(self):
        return hash((self.name, self.startEntity, self.endEntity))

    def __repr__(self):
        return f"Relationship(name={self.name}, startEntity={self.startEntity}, endEntity={self.endEntity}, properties={self.properties}, properties_info={self.properties_info})"


class KnowledgeGraph(BaseModel):
    entities:list[Entity]= []
    relationships:list[Relationship] = []

    def embed_entities(self,
                       embeddings_function:Callable[[str], np.array],
                       entity_name_weight:float=0.6,
                       entity_label_weight:float=0.4)-> None:
        self.remove_duplicates_entities()
        for Entity in self.entities:
            Entity.process()
        entities_embeddings = (
            entity_label_weight * embeddings_function([Entity.label for Entity in self.entities])
            +
            entity_name_weight * embeddings_function([Entity.name for Entity in self.entities])
            )

        for Entity, embedding in zip(self.entities, entities_embeddings):
            Entity.properties.embeddings = embedding


    def embed_relationships(self, embeddings_function:Callable[[str], np.array])-> None:
        self.remove_duplicates_relationships()
        for relationship in self.relationships:
            relationship.process()

        relationships_embeddings = (
            embeddings_function([relationship.name for relationship in self.relationships])
            )

        for relationship, embedding in zip(self.relationships, relationships_embeddings):
            relationship.properties.embeddings = embedding

    def get_entity(self, other_entity:Entity):
        for entity in self.entities:
            if entity == other_entity :
                return entity
        return None

    def remove_duplicates_entities(self) -> None:
        """
        Remove duplicate entities (entities) by relying on the `__hash__` and `__eq__` methods of the `Entity` class.
        This will update the `entities` attribute by filtering out duplicates.
        """

        self.entities = list(dict.fromkeys(self.entities))  # Stable order for trace replay.

    def remove_duplicates_relationships(self) -> None:
        """
        Remove duplicate relationships by relying on the `__hash__` and `__eq__` methods of the `Relationship` class.
        This will update the `relationships` attribute by filtering out duplicates.
        """
        all_relationship = self.relationships
        relationship_list = []
        output_relationship = []

        for relationship in all_relationship:
            relationship_tuple_1 = (relationship.startEntity.name, relationship.endEntity.name, relationship.name)
            relationship_tuple_2 = (relationship.endEntity.name, relationship.startEntity.name, relationship.name)
            if relationship_tuple_1 not in relationship_list and relationship_tuple_2 not in relationship_list:
                relationship_list.append(relationship_tuple_1)
                relationship_list.append(relationship_tuple_2)
                output_relationship.append(relationship)
        self.relationships = output_relationship

       # self.relationships = list(set(self.relationships))  # Using set to automatically remove duplicates based on hash and eq methods

    def find_isolated_entities(self):
        relation_entities = set(rel.startEntity for rel in self.relationships) | set(rel.endEntity for rel in self.relationships)
        isolated_entities = [ent for ent in self.entities if ent not in relation_entities]
        return isolated_entities

    def remove_isolated_entities(self):
        isolated_entities = self.find_isolated_entities()
        logging.info(f"INFO ---- Removing isolated entities {[i.name for i in isolated_entities]}")
        self.entities = [ent for ent in self.entities if ent not in isolated_entities]
