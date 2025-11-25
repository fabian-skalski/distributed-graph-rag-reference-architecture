"""Graph processing business logic."""
import logging
from typing import List, Dict, Any, Tuple

import igraph as ig

logger = logging.getLogger(__name__)


class GraphService:
    """Service for graph building and community detection.
    
    Provides methods for parsing summaries to extract entities and relationships,
    building knowledge graphs, detecting communities, and generating community
    descriptions.
    """
    
    def __init__(self):
        """Initialize graph service with empty graph instance."""
        self.current_graph: ig.Graph = None
    
    @staticmethod
    def parse_summary(summary: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Parse a summary to extract entities and relationships.
        
        Args:
            summary: Text summary containing entities and relationships.
            
        Returns:
            Tuple[List[str], List[Tuple[str, str, str]]]: Tuple containing a list
                of entities and a list of relationship tuples (source, relation, target).
        """
        entities = []
        relationships = []
        
        lines = summary.split("\n")
        is_entities_section = False
        is_relationships_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            line_lower = line.lower()
            if "entities:" in line_lower or (line_lower == "entities" and not is_relationships_section):
                is_entities_section = True
                is_relationships_section = False
                continue
            elif "relationships:" in line_lower or line_lower == "relationships":
                is_entities_section = False
                is_relationships_section = True
                continue
            
            # Skip JSON-like structures
            if line.startswith(("{", "}", "[", "]")) or any(
                key in line for key in ['"id":', '"name":', '"type":', '"attributes":']
            ):
                continue
            
            if is_entities_section:
                entity = line.lstrip("0123456789.-*• ").replace("**", "").strip()
                if entity and not any(
                    prefix in entity for prefix in ["name:", "type:", "attributes:", "popularity:"]
                ):
                    entities.append(entity)
            
            elif is_relationships_section and "->" in line:
                parts = [p.strip() for p in line.split("->")]
                if len(parts) >= 3:
                    source = parts[0].lstrip("-*• ").strip()
                    relation = parts[1].strip()
                    target = parts[2].strip()
                    if source and target:
                        relationships.append((source, relation, target))
        
        return entities, relationships
    
    def build_graph(self, summaries: List[str]) -> Dict[str, Any]:
        """Build knowledge graph from summaries.
        
        Args:
            summaries: List of text summaries containing entities and relationships.
            
        Returns:
            Dict[str, Any]: Dictionary with graph statistics including node and edge counts.
        """
        logger.info(f"Building graph from {len(summaries)} summaries")
        
        vertices = set()
        edges = []
        edge_labels = []
        
        for idx, summary in enumerate(summaries):
            entities, relationships = self.parse_summary(summary)
            
            # Add entities as vertices
            for entity in entities:
                vertices.add(entity)
            
            # Add relationships as edges
            for source, relation, target in relationships:
                vertices.add(source)
                vertices.add(target)
                edges.append((source, target))
                edge_labels.append(relation)
        
        # Create graph
        graph = ig.Graph()
        vertex_list = list(vertices)
        graph.add_vertices(vertex_list)
        
        if edges:
            graph.add_edges(edges)
            graph.es["label"] = edge_labels
        
        self.current_graph = graph
        
        logger.info(f"Graph built with {graph.vcount()} nodes and {graph.ecount()} edges")
        
        return {
            "nodes": graph.vcount(),
            "edges": graph.ecount()
        }
    
    def detect_communities(self, min_community_size: int = 3, resolution: float = 1.0) -> List[Dict[str, Any]]:
        """Detect communities using Leiden algorithm with smarter merging.
        
        Args:
            min_community_size: Minimum size for a community (smaller ones get merged).
            resolution: Resolution parameter for Leiden (lower = fewer, larger communities).
        
        Returns:
            List[Dict[str, Any]]: List of community dictionaries containing community_id,
                members list, and size.
                
        Raises:
            ValueError: If no graph is available (must call build_graph first).
        """
        if self.current_graph is None:
            raise ValueError("No graph available. Build graph first.")
        
        logger.info(f"Detecting communities (min_size={min_community_size}, resolution={resolution})")
        communities = []
        
        # For small graphs, use a single community or simple partitioning
        if self.current_graph.vcount() < 10:
            logger.info("Small graph detected, using single community")
            return [{
                "community_id": 0,
                "members": [v["name"] for v in self.current_graph.vs],
                "size": self.current_graph.vcount()
            }]
        
        # For larger graphs, use Leiden with lower resolution for fewer communities
        try:
            partition = self.current_graph.community_leiden(
                objective_function='modularity',
                resolution_parameter=resolution,
                n_iterations=3  # Limit iterations for speed
            )
            
            # Group communities and merge small ones
            temp_communities = []
            small_members = []
            
            for community_indices in partition:
                members = [self.current_graph.vs[i]["name"] for i in community_indices]
                if len(members) >= min_community_size:
                    temp_communities.append(members)
                else:
                    small_members.extend(members)
            
            # Merge all small communities into one if they exist
            if small_members:
                temp_communities.append(small_members)
            
            # Create final community list
            for idx, members in enumerate(temp_communities):
                communities.append({
                    "community_id": idx,
                    "members": members,
                    "size": len(members)
                })
            
        except Exception as e:
            logger.exception("Error in community detection, using single community fallback")
            # Fallback: treat entire graph as one community
            communities = [{
                "community_id": 0,
                "members": [v["name"] for v in self.current_graph.vs],
                "size": self.current_graph.vcount()
            }]
        
        logger.info(f"Detected {len(communities)} communities")
        return communities
    
    def get_community_description(self, community_members: List[str]) -> Dict[str, Any]:
        """Get description of a community.
        
        Args:
            community_members: List of entity names in the community.
            
        Returns:
            Dict[str, Any]: Dictionary containing entities list and relationships list
                for the specified community.
                
        Raises:
            ValueError: If no graph is available (must call build_graph first).
        """
        if self.current_graph is None:
            raise ValueError("No graph available. Build graph first.")
        
        try:
            vertex_indices = [
                self.current_graph.vs.find(name=node).index
                for node in community_members
            ]
            subgraph = self.current_graph.subgraph(vertex_indices)
            
            entities = [subgraph.vs[i]["name"] for i in range(subgraph.vcount())]
            relationships = []
            
            for edge in subgraph.es:
                source = subgraph.vs[edge.source]['name']
                target = subgraph.vs[edge.target]['name']
                label = edge['label'] if 'label' in edge.attributes() else 'related_to'
                relationships.append(f"{source} -> {label} -> {target}")
            
            return {
                'entities': entities,
                'relationships': relationships
            }
        except Exception as e:
            logger.exception("Error getting community description")
            return {
                'entities': community_members,
                'relationships': []
            }
