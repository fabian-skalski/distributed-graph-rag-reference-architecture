"""Neo4j database service for document caching and graph storage."""
import logging
import hashlib
import json
from typing import Optional, Dict, List, Any
from neo4j import GraphDatabase, Driver
from pydantic_settings import BaseSettings
from pydantic import Field


logger = logging.getLogger(__name__)


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings."""
    uri: str = Field(alias="NEO4J_URI")
    username: str = Field(alias="NEO4J_USERNAME")
    password: str = Field(alias="NEO4J_PASSWORD")
    database: str = Field(alias="NEO4J_DATABASE")

    class Config:
        env_file = ".env"
        case_sensitive = False


class Neo4jService:
    """Service for interacting with Neo4j graph database."""
    
    def __init__(self, settings: Optional[Neo4jSettings] = None):
        """Initialize Neo4j connection."""
        self.settings = settings or Neo4jSettings()
        self.driver: Optional[Driver] = None
        
    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.username, self.settings.password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                keep_alive=True
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.settings.uri}")
            
            # Create indexes and constraints
            self._create_schema()
            
        except Exception as e:
            logger.exception("Failed to connect to Neo4j")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def _create_schema(self):
        """Create necessary indexes and constraints."""
        with self.driver.session(database=self.settings.database) as session:
            # Constraints
            session.run(
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.document_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )
            
            # Indexes for faster lookups
            session.run(
                "CREATE INDEX document_file_path IF NOT EXISTS "
                "FOR (d:Document) ON (d.file_path)"
            )
            session.run(
                "CREATE INDEX chunk_document_id IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.document_id)"
            )
            
            logger.info("Neo4j schema created/verified")
    
    def check_document_cached(self, file_path: str, content: str) -> tuple[bool, Optional[str]]:
        """Check if document is already cached."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (d:Document {file_path: $file_path, content_hash: $content_hash})
                RETURN d.document_id as document_id
                """,
                file_path=file_path,
                content_hash=content_hash
            )
            record = result.single()
            
            if record:
                return True, record["document_id"]
            return False, None
    
    def save_document(self, document_id: str, file_path: str, content: str, metadata: Optional[Dict] = None):
        """Save document to Neo4j."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Convert metadata to JSON string for Neo4j storage
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        with self.driver.session(database=self.settings.database) as session:
            session.run(
                """
                MERGE (d:Document {document_id: $document_id})
                SET d.file_path = $file_path,
                    d.content = $content,
                    d.content_hash = $content_hash,
                    d.metadata = $metadata_json,
                    d.created_at = datetime()
                """,
                document_id=document_id,
                file_path=file_path,
                content=content,
                content_hash=content_hash,
                metadata_json=metadata_json
            )
            logger.info(f"Document saved: {document_id}")
    
    def save_chunks(self, document_id: str, chunks: List[Dict[str, Any]]):
        """Save document chunks to Neo4j."""
        with self.driver.session(database=self.settings.database) as session:
            for chunk in chunks:
                chunk_id = f"{document_id}_chunk_{chunk.get('chunk_index', 0)}"
                session.run(
                    """
                    MATCH (d:Document {document_id: $document_id})
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.content = $content,
                        c.chunk_index = $chunk_index,
                        c.start_char = $start_char,
                        c.end_char = $end_char,
                        c.document_id = $document_id
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    content=chunk.get("content", ""),
                    chunk_index=chunk.get("chunk_index", 0),
                    start_char=chunk.get("start_char", 0),
                    end_char=chunk.get("end_char", 0)
                )
            logger.info(f"Saved {len(chunks)} chunks for document {document_id}")
    
    def get_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve chunks for a document."""
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.chunk_id as chunk_id,
                       c.content as content,
                       c.chunk_index as chunk_index,
                       c.start_char as start_char,
                       c.end_char as end_char
                ORDER BY c.chunk_index
                """,
                document_id=document_id
            )
            
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "content": record["content"],
                    "chunk_index": record["chunk_index"],
                    "start_char": record["start_char"],
                    "end_char": record["end_char"]
                })
            
            return chunks
    
    def save_elements(self, document_id: str, elements: List[Dict[str, Any]]):
        """Save extracted elements to Neo4j."""
        with self.driver.session(database=self.settings.database) as session:
            for idx, element in enumerate(elements):
                element_id = f"{document_id}_element_{idx}"
                session.run(
                    """
                    MATCH (d:Document {document_id: $document_id})
                    MERGE (e:Element {element_id: $element_id})
                    SET e.content = $content,
                        e.chunk_index = $chunk_index,
                        e.element_index = $element_index,
                        e.document_id = $document_id
                    MERGE (d)-[:HAS_ELEMENT]->(e)
                    """,
                    document_id=document_id,
                    element_id=element_id,
                    content=element.get("content", ""),
                    chunk_index=element.get("chunk_index", idx),
                    element_index=idx
                )
            logger.info(f"Saved {len(elements)} elements for document {document_id}")
    
    def save_summaries(self, document_id: str, summaries: List[Dict[str, Any]]):
        """Save element summaries to Neo4j."""
        with self.driver.session(database=self.settings.database) as session:
            for idx, summary in enumerate(summaries):
                summary_id = f"{document_id}_summary_{idx}"
                session.run(
                    """
                    MATCH (d:Document {document_id: $document_id})
                    MERGE (s:Summary {summary_id: $summary_id})
                    SET s.summary = $summary,
                        s.element_id = $element_id,
                        s.summary_index = $summary_index,
                        s.document_id = $document_id
                    MERGE (d)-[:HAS_SUMMARY]->(s)
                    """,
                    document_id=document_id,
                    summary_id=summary_id,
                    summary=summary.get("summary", ""),
                    element_id=summary.get("element_id", idx),
                    summary_index=idx
                )
            logger.info(f"Saved {len(summaries)} summaries for document {document_id}")
    
    def get_summaries(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve summaries for a document."""
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (d:Document {document_id: $document_id})-[:HAS_SUMMARY]->(s:Summary)
                RETURN s.summary_id as summary_id,
                       s.summary as summary,
                       s.element_id as element_id,
                       s.summary_index as summary_index
                ORDER BY s.summary_index
                """,
                document_id=document_id
            )
            
            summaries = []
            for record in result:
                summaries.append({
                    "summary_id": record["summary_id"],
                    "summary": record["summary"],
                    "element_id": record["element_id"],
                    "summary_index": record["summary_index"]
                })
            
            return summaries
    
    def save_graph(self, summaries_hash: str, nodes: int, edges: int, communities: List[Dict[str, Any]]):
        """Save graph structure and communities to Neo4j.
        
        Args:
            summaries_hash: Hash of the summaries used to build the graph
            nodes: Number of nodes in the graph
            edges: Number of edges in the graph
            communities: List of detected communities
        """
        with self.driver.session(database=self.settings.database) as session:
            # Save graph metadata
            session.run(
                """
                MERGE (g:GraphCache {summaries_hash: $summaries_hash})
                SET g.nodes = $nodes,
                    g.edges = $edges,
                    g.communities_count = $communities_count,
                    g.updated_at = datetime()
                """,
                summaries_hash=summaries_hash,
                nodes=nodes,
                edges=edges,
                communities_count=len(communities)
            )
            
            # Save communities
            for community in communities:
                session.run(
                    """
                    MATCH (g:GraphCache {summaries_hash: $summaries_hash})
                    MERGE (c:Community {
                        summaries_hash: $summaries_hash, 
                        community_id: $community_id
                    })
                    SET c.members = $members,
                        c.size = $size
                    MERGE (g)-[:HAS_COMMUNITY]->(c)
                    """,
                    summaries_hash=summaries_hash,
                    community_id=community["community_id"],
                    members=community["members"],
                    size=community["size"]
                )
            
            logger.info(f"Saved graph with {nodes} nodes, {edges} edges, {len(communities)} communities")
    
    def get_graph(self, summaries_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached graph structure.
        
        Args:
            summaries_hash: Hash of the summaries
            
        Returns:
            Graph data if found, None otherwise
        """
        with self.driver.session(database=self.settings.database) as session:
            # Get graph metadata
            graph_result = session.run(
                """
                MATCH (g:GraphCache {summaries_hash: $summaries_hash})
                RETURN g.nodes as nodes, 
                       g.edges as edges,
                       g.communities_count as communities_count
                """,
                summaries_hash=summaries_hash
            )
            graph_record = graph_result.single()
            
            if not graph_record:
                return None
            
            # Get communities
            communities_result = session.run(
                """
                MATCH (g:GraphCache {summaries_hash: $summaries_hash})-[:HAS_COMMUNITY]->(c:Community)
                RETURN c.community_id as community_id,
                       c.members as members,
                       c.size as size
                ORDER BY c.community_id
                """,
                summaries_hash=summaries_hash
            )
            
            communities = []
            for record in communities_result:
                communities.append({
                    "community_id": record["community_id"],
                    "members": record["members"],
                    "size": record["size"]
                })
            
            return {
                "nodes": graph_record["nodes"],
                "edges": graph_record["edges"],
                "communities": communities
            }
    
    def save_community_descriptions(self, summaries_hash: str, descriptions: List[Dict[str, Any]]):
        """Save community descriptions to Neo4j.
        
        Args:
            summaries_hash: Hash of the summaries
            descriptions: List of community descriptions
        """
        with self.driver.session(database=self.settings.database) as session:
            for idx, description in enumerate(descriptions):
                session.run(
                    """
                    MATCH (g:GraphCache {summaries_hash: $summaries_hash})
                    MERGE (cd:CommunityDescription {
                        summaries_hash: $summaries_hash,
                        description_index: $idx
                    })
                    SET cd.entities = $entities,
                        cd.relationships = $relationships
                    MERGE (g)-[:HAS_DESCRIPTION]->(cd)
                    """,
                    summaries_hash=summaries_hash,
                    idx=idx,
                    entities=description.get("entities", []),
                    relationships=description.get("relationships", [])
                )
            
            logger.info(f"Saved {len(descriptions)} community descriptions")
    
    def get_community_descriptions(self, summaries_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached community descriptions.
        
        Args:
            summaries_hash: Hash of the summaries
            
        Returns:
            List of descriptions if found, None otherwise
        """
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (g:GraphCache {summaries_hash: $summaries_hash})-[:HAS_DESCRIPTION]->(cd:CommunityDescription)
                RETURN cd.entities as entities,
                       cd.relationships as relationships,
                       cd.description_index as idx
                ORDER BY cd.description_index
                """,
                summaries_hash=summaries_hash
            )
            
            descriptions = []
            for record in result:
                descriptions.append({
                    "entities": record["entities"],
                    "relationships": record["relationships"]
                })
            
            return descriptions if descriptions else None
    
    def save_community_summaries(self, summaries_hash: str, summaries: List[str]):
        """Save community summaries to Neo4j.
        
        Args:
            summaries_hash: Hash of the summaries
            summaries: List of community summaries
        """
        with self.driver.session(database=self.settings.database) as session:
            for idx, summary in enumerate(summaries):
                session.run(
                    """
                    MATCH (g:GraphCache {summaries_hash: $summaries_hash})
                    MERGE (cs:CommunitySummary {
                        summaries_hash: $summaries_hash,
                        summary_index: $idx
                    })
                    SET cs.summary = $summary
                    MERGE (g)-[:HAS_COMMUNITY_SUMMARY]->(cs)
                    """,
                    summaries_hash=summaries_hash,
                    idx=idx,
                    summary=summary
                )
            
            logger.info(f"Saved {len(summaries)} community summaries")
    
    def get_community_summaries(self, summaries_hash: str) -> Optional[List[str]]:
        """Retrieve cached community summaries.
        
        Args:
            summaries_hash: Hash of the summaries
            
        Returns:
            List of summaries if found, None otherwise
        """
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (g:GraphCache {summaries_hash: $summaries_hash})-[:HAS_COMMUNITY_SUMMARY]->(cs:CommunitySummary)
                RETURN cs.summary as summary,
                       cs.summary_index as idx
                ORDER BY cs.summary_index
                """,
                summaries_hash=summaries_hash
            )
            
            summaries = []
            for record in result:
                summaries.append(record["summary"])
            
            return summaries if summaries else None
    
    def save_query_answer(self, query_hash: str, answer: str):
        """Save a query answer to cache.
        
        Args:
            query_hash: Hash of query+documents
            answer: The answer to cache
        """
        with self.driver.session(database=self.settings.database) as session:
            session.run(
                """
                MERGE (qa:QueryAnswer {query_hash: $query_hash})
                SET qa.answer = $answer,
                    qa.cached_at = datetime()
                """,
                query_hash=query_hash,
                answer=answer
            )
            logger.info(f"Saved query answer for hash {query_hash[:8]}...")
    
    def get_query_answer(self, query_hash: str) -> Optional[str]:
        """Retrieve cached query answer.
        
        Args:
            query_hash: Hash of query+documents
            
        Returns:
            Cached answer if found, None otherwise
        """
        with self.driver.session(database=self.settings.database) as session:
            result = session.run(
                """
                MATCH (qa:QueryAnswer {query_hash: $query_hash})
                RETURN qa.answer as answer
                """,
                query_hash=query_hash
            )
            
            record = result.single()
            return record["answer"] if record else None
    
    def clear_all(self):
        """Clear all data from Neo4j database."""
        with self.driver.session(database=self.settings.database) as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared all data from Neo4j")
            
            # Recreate schema
            self._create_schema()
    
    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            if self.driver:
                self.driver.verify_connectivity()
                return True
            return False
        except Exception as e:
            logger.exception("Health check failed")
            return False
