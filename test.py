# Simple parallel implementation for https://arxiv.org/abs/2404.16130
from openai import OpenAI
import igraph as ig
import os
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load documents from file
def load_documents(folder="test_docs"):
    """Load all .txt files from a folder, where each file is one document."""
    documents = []
    try:
        txt_files = sorted([f for f in os.listdir(folder) if f.endswith('.txt')])
        
        if not txt_files:
            logger.warning(f"No .txt files found in {folder}")
            return documents
        
        for filename in txt_files:
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
                    logger.info(f"Loaded document from {filename}: {len(content)} chars")
        
        logger.info(f"Loaded {len(documents)} documents from {folder}")
        return documents
    except FileNotFoundError:
        logger.error(f"Folder not found: {folder}")
        raise
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

DOCUMENTS = load_documents()

# Parse query parameters from environment variable
query_params_str = os.getenv("OPENAI_QUERY_PARAMS", "")
default_query = None
if query_params_str:
    # Convert string like "api-version=2023-07-01-preview" to dict
    default_query = {}
    for param in query_params_str.split("&"):
        if "=" in param:
            key, value = param.split("=", 1)
            default_query[key] = value

# Initialize OpenAI client with custom base URL and default query parameters
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),  # Custom OpenAI API endpoint
    default_query=default_query,  # Custom query parameters as dict
)

logger.info(f"OpenAI client initialized with base_url: {os.getenv('OPENAI_BASE_URL')}")
logger.info(f"Query parameters: {default_query}")
logger.info(f"Using model: {os.getenv('OPENAI_INFERENCE_MODEL_NAME')}")

# Configuration for parallel processing
MAX_WORKERS = 10  # Number of parallel API calls
CHUNK_SIZE = 600  # Default chunk size for document splitting
CHUNK_OVERLAP = 100  # Default overlap size between chunks, cf. https://arxiv.org/abs/2404.16130


def call_openai_api(model_name, messages, request_type=""):
    """Unified function to call OpenAI API with error handling and logging."""
    logger.debug(f"API Request [{request_type}] - Model: {model_name}")
    logger.debug(f"API Request [{request_type}] - Messages: {json.dumps(messages, indent=2)}")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        
        logger.info(f"API Response [{request_type}] - Status: Success")
        logger.debug(f"API Response [{request_type}] - Usage: {response.usage}")
        logger.debug(f"API Response [{request_type}] - Content: {response.choices[0].message.content}")
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"API Request [{request_type}] failed: {str(e)}", exc_info=True)
        raise


# 1. Source Documents → Text Chunks
def chunk_documents(documents):
    """Split documents into overlapping chunks using centralized config."""
    chunks = []
    
    def process_document(document):
        doc_chunks = []
        for i in range(0, len(document), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = document[i:i + CHUNK_SIZE]
            doc_chunks.append(chunk)
        return doc_chunks
    
    # Parallel processing for large document sets
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(process_document, documents)
        for doc_chunks in results:
            chunks.extend(doc_chunks)
    
    return chunks


# 2. Text Chunks → Element Instances (Parallelized)
def extract_single_chunk(index, chunk, model_name):
    """Extract elements from a single chunk."""
    logger.info(f"Processing chunk {index + 1}")
    
    messages = [
        {"role": "system", "content": "Extract entities and relationships from the following text."},
        {"role": "user", "content": chunk}
    ]
    
    content = call_openai_api(model_name, messages, f"extract_chunk_{index}")
    print(f"Chunk {index}: Extracted {len(content)} chars")
    return content


def extract_elements(chunks):
    """Extract elements from all chunks in parallel."""
    model_name = os.getenv("OPENAI_INFERENCE_MODEL_NAME")
    elements = [None] * len(chunks)  # Pre-allocate list
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(extract_single_chunk, i, chunk, model_name): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            elements[index] = future.result()
    
    return elements


# 3. Element Instances → Element Summaries (Parallelized)
def summarize_single_element(index, element, model_name):
    """Summarize a single element."""
    logger.info(f"Summarizing element {index + 1}")
    
    messages = [
        {"role": "system", "content": """Summarize the extracted entities and relationships in a clear, structured format.

Output format:
Entities:
- Entity1
- Entity2
- Entity3

Relationships:
Entity1 -> relationship_type -> Entity2
Entity3 -> relationship_type -> Entity1

Guidelines:
- List each entity on a separate line with a bullet point or number
- Use arrow notation (A -> relation -> B) for relationships
- Keep entity names consistent throughout
- Include only factual information from the input"""},
        {"role": "user", "content": element}
    ]
    
    content = call_openai_api(model_name, messages, f"summarize_element_{index}")
    print(f"Element {index}: Summarized")
    return content


def create_summaries(elements):
    """Summarize all elements in parallel."""
    model_name = os.getenv("OPENAI_INFERENCE_MODEL_NAME")
    summaries = [None] * len(elements)  # Pre-allocate list
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(summarize_single_element, i, element, model_name): i 
            for i, element in enumerate(elements)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            summaries[index] = future.result()
    
    return summaries


# 4. Element Summaries → Graph Communities
def build_graph(summaries):
    """Build knowledge graph from element summaries."""
    logger.info(f"Building graph from {len(summaries)} summaries")
    
    vertices = set()
    edges = []
    edge_labels = []
    
    for index, summary in enumerate(summaries):
        print(f"Summary index {index} of {len(summaries)}:")
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
            if line.startswith(("{", "}", "[", "]")) or any(key in line for key in ['"id":', '"name":', '"type":', '"attributes":']):
                continue
            
            if is_entities_section:
                # Extract entity name from list items
                entity = line.lstrip("0123456789.-*• ").replace("**", "").strip()
                if entity and not any(prefix in entity for prefix in ["name:", "type:", "attributes:", "popularity:"]):
                    vertices.add(entity)
                    logger.debug(f"Added node: {entity}")
                    
            elif is_relationships_section and "->" in line:
                # Parse relationship: "A -> relation -> B"
                parts = [p.strip() for p in line.split("->")]
                if len(parts) >= 3:
                    source = parts[0].lstrip("-*• ").strip()
                    relation = parts[1].strip()
                    target = parts[2].strip()
                    if source and target:
                        vertices.add(source)
                        vertices.add(target)
                        edges.append((source, target))
                        edge_labels.append(relation)
                        logger.debug(f"Added edge: {source} -> {relation} -> {target}")
    
    # Create graph
    G = ig.Graph()
    G.add_vertices(list(vertices))
    if edges:
        G.add_edges(edges)
        G.es["label"] = edge_labels
    
    logger.info(f"Graph built with {G.vcount()} nodes and {G.ecount()} edges")
    return G


# 5. Graph Communities → Community Summaries
def find_communities(graph):
    """Detect communities using Leiden algorithm."""
    logger.info("Detecting communities in graph")
    communities = []
    
    # Process each connected component
    for idx, component in enumerate(graph.connected_components(mode="weak")):
        print(f"Component index {idx} of {len(graph.connected_components(mode='weak'))}:")
        subgraph = graph.subgraph(component)
        
        if subgraph.vcount() > 1:
            try:
                # Detect communities within component using Leiden algorithm
                partition = subgraph.community_leiden(objective_function='modularity')
                for community_indices in partition:
                    community = [subgraph.vs[i]["name"] for i in community_indices]
                    communities.append(community)
                    logger.debug(f"Found community with {len(community)} nodes")
            except Exception as e:
                logger.error(f"Error in community detection for component {idx}: {e}")
                # Fallback: treat whole component as one community
                communities.append([subgraph.vs[i]["name"] for i in range(subgraph.vcount())])
        else:
            # Single-node community
            communities.append([subgraph.vs[0]["name"]])
    
    logger.info(f"Detected {len(communities)} communities")
    print("Communities from find_communities:", communities)
    return communities


def summarize_community(index, community, graph, model_name):
    """Generate a summary for a single community."""
    logger.info(f"Summarizing community {index + 1}")
    
    # Get subgraph and build description
    vertex_indices = [graph.vs.find(name=node).index for node in community]
    subgraph = graph.subgraph(vertex_indices)
    
    entities = [subgraph.vs[i]["name"] for i in range(subgraph.vcount())]
    relationships = [
        f"{subgraph.vs[e.source]['name']} -> {e['label'] if 'label' in e.attributes() else 'related_to'} -> {subgraph.vs[e.target]['name']}"
        for e in subgraph.es
    ]
    
    description = f"Entities: {', '.join(entities)}\nRelationships: {', '.join(relationships)}"
    logger.debug(f"Community description: {description}")
    
    messages = [
        {"role": "system", "content": "Summarize the following community of entities and relationships."},
        {"role": "user", "content": description}
    ]
    
    return call_openai_api(model_name, messages, f"summarize_community_{index}")


def create_community_summaries(communities, graph):
    """Summarize all communities in parallel."""
    logger.info(f"Summarizing {len(communities)} communities")
    
    if not communities:
        return []
    
    model_name = os.getenv("OPENAI_INFERENCE_MODEL_NAME")
    community_summaries = [None] * len(communities)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(summarize_community, i, community, graph, model_name): i 
            for i, community in enumerate(communities)
        }
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            community_summaries[index] = future.result()
    
    return community_summaries


# 6. Community Summaries → Community Answers → Global Answer (Parallelized)
def answer_from_summary(index, summary, query, model_name):
    """Generate answer from a single community summary."""
    logger.info(f"Processing community summary {index + 1}")
    
    messages = [
        {"role": "system", "content": "Answer the following query based on the provided summary."},
        {"role": "user", "content": f"Query: {query} Summary: {summary}"}
    ]
    
    return call_openai_api(model_name, messages, f"answer_community_{index}")


def generate_final_answer(community_summaries, query):
    """Generate answers from all communities in parallel and combine them."""
    logger.info(f"Generating answers from {len(community_summaries)} community summaries")
    
    if not community_summaries:
        return "No communities found to analyze."
    
    model_name = os.getenv("OPENAI_INFERENCE_MODEL_NAME")
    intermediate_answers = [None] * len(community_summaries)
    
    # Parallel processing of community summaries
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(answer_from_summary, i, summary, query, model_name): i 
            for i, summary in enumerate(community_summaries)
        }
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            intermediate_answers[index] = future.result()
            print(f"Intermediate answer {index + 1}: Generated")

    # Generate final answer
    logger.info("Generating final answer from intermediate answers")
    final_messages = [
        {"role": "system", "content": "Combine these answers into a final, concise response."},
        {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
    ]
    
    final_answer = call_openai_api(model_name, final_messages, "final_answer")
    return final_answer


# Putting It All Together
def run_pipeline(documents, query):
    """Execute the complete Graph RAG pipeline."""
    import time
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("Starting Graph RAG Pipeline (OPTIMIZED)")
    logger.info(f"Query: {query}")
    logger.info(f"Number of documents: {len(documents)}")
    logger.info(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    logger.info(f"Max parallel workers: {MAX_WORKERS}")
    logger.info("=" * 80)
    
    # Step 1: Split documents into chunks
    step_start = time.time()
    logger.info("Step 1: Splitting documents into chunks")
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks in {time.time() - step_start:.2f}s")

    # Step 2: Extract elements from chunks (PARALLEL)
    step_start = time.time()
    logger.info("Step 2: Extracting elements from chunks (PARALLEL)")
    elements = extract_elements(chunks)
    logger.info(f"Extracted {len(elements)} element sets in {time.time() - step_start:.2f}s")

    # Step 3: Summarize elements (PARALLEL)
    step_start = time.time()
    logger.info("Step 3: Summarizing elements (PARALLEL)")
    summaries = create_summaries(elements)
    logger.info(f"Created {len(summaries)} summaries in {time.time() - step_start:.2f}s")

    # Step 4: Build graph and detect communities
    step_start = time.time()
    logger.info("Step 4: Building graph and detecting communities")
    graph = build_graph(summaries)
    print(f"Graph: {graph.vcount()} nodes, {graph.ecount()} edges")
    communities = find_communities(graph)
    logger.info(f"Detected {len(communities)} communities in {time.time() - step_start:.2f}s")

    if communities:
        print(f"First community: {communities[0]}")
    else:
        logger.warning("No communities detected. Graph may be empty or disconnected.")
        print("communities: []")
        
    # Step 5: Summarize communities (PARALLEL)
    step_start = time.time()
    logger.info("Step 5: Summarizing communities (PARALLEL)")
    community_summaries = create_community_summaries(communities, graph)
    logger.info(f"Created {len(community_summaries)} community summaries in {time.time() - step_start:.2f}s")

    # Step 6: Generate answers from community summaries (PARALLEL)
    step_start = time.time()
    logger.info("Step 6: Generating final answer (PARALLEL)")
    final_answer = generate_final_answer(community_summaries, query)
    logger.info(f"Generated final answer in {time.time() - step_start:.2f}s")

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Graph RAG Pipeline completed successfully in {total_time:.2f}s")
    logger.info("=" * 80)
    
    return final_answer


# Example usage
query = "What are the main themes in these documents?"
print('Query:', query)
answer = run_pipeline(DOCUMENTS, query)
print('Answer:', answer)