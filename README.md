# distributed-graph-rag-reference-architecture

## Introduction

**distributed-graph-rag-reference-architecture** is an enterprise-grade reference implementation designed to help organizations understand, customize, and deploy large-scale GraphRAG-based query engines. This project demonstrates how to architect a distributed knowledge graph system that transforms unstructured documents into queryable insights using advanced retrieval-augmented generation (RAG) techniques.

**Why microservices?** At scale, different pipeline stages have conflicting resource needs: graph processing (igraph community detection) is CPU-bound requiring compute-optimized instances, LLM calls are network I/O bound (or GPU-bound for self-hosted models) best handled by standard or GPU instances, all operations run asynchronously to handle high concurrency, specialized data stores optimize different access patterns (Neo4j for graph queries with vector support, Redis for distributed rate limiting), etc. Decoupling these concerns enables independent scaling and cost optimization.

### AI Research Foundation

This GraphRAG implementation is based on the methodology described in **[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)** (Microsoft Research, April 2024). The paper introduces a novel approach to RAG that builds hierarchical knowledge graphs from text corpora, enabling sophisticated global reasoning across entire document collections rather than simple local context retrieval.

### Value Proposition: Why This Architecture vs. Established Libraries?

While libraries like `microsoft/graphrag` provide turnkey solutions, enterprise adoption often requires deeper architectural understanding and customization. This reference architecture offers distinct advantages:

**🔧 Architectural Transparency**
- **Full visibility into every component**: Unlike monolithic libraries, each microservice is independently documented and can be studied, modified, or replaced. Teams can understand exactly how documents flow through the pipeline, how graphs are constructed, and how queries are resolved.
- **Educational foundation**: Perfect for teams building internal ML/AI capabilities who need to understand the mechanics before committing to production.

**🌐 Distributed Scale & Resilience**
- **Microservices architecture**: Each component (document processing, LLM operations, graph building, caching, rate limiting) runs as an independent service. This enables:
  - Horizontal scaling of bottleneck components
  - Independent deployment and rollback
  - Fault isolation (one service failure doesn't crash the system)
  - Technology heterogeneity (swap Python services for Go/Rust if needed)
- **Enterprise-ready infrastructure**: Built-in support for distributed rate limiting (Redis), persistent graph storage (Neo4j), centralized logging (Fluentd), and containerized deployment (Docker Compose/Kubernetes-ready).

**🎨 Customizability Without Vendor Lock-In**
- **Domain-specific optimization**: Easily modify extraction prompts, chunking strategies, community detection algorithms, or graph schemas to match your industry's needs (legal, financial, healthcare, etc.).
- **LLM flexibility**: Not locked into a single provider—swap OpenAI for Anthropic, Azure OpenAI, or local LLMs (Llama, Mixtral) by changing configuration.
- **Data sovereignty**: Complete control over where and how your data is processed. Deploy on-premises or in your cloud environment without third-party dependencies.

**💡 Production-Grade Patterns**
- **Intelligent caching**: Multi-level caching strategy (documents, chunks, graph structures, query answers) minimizes expensive LLM calls.
- **Rate limiting**: Token bucket algorithm prevents quota exhaustion across distributed workers.
- **Observability**: Structured logging, health checks, and monitoring hooks built-in from day one.

**When to use this architecture:**
- Scaling beyond single-machine limits (millions of documents, thousands of concurrent queries)
- Integrating GraphRAG into existing microservices ecosystems
- Need full control over data processing pipelines for compliance/security
- Building a proof-of-concept to understand GraphRAG before committing to a vendor
- Customizing graph schemas, entity extraction, or community detection for specialized domains

**When to use a library instead:**
- Rapid prototyping with small datasets (<1000 documents)
- Research experiments where implementation details don't matter
- Acceptable to rely on vendor-maintained abstractions

---

## Quick Start

**Prerequisites:** Docker, Docker Compose, and OpenAI API key. Create a `.env` file with `OPENAI_API_KEY=your-key-here` (see `.env.example`). Start services with `docker-compose up -d`, then run `python simple_query_app.py` to interactively query documents. The system indexes `.txt` files from the `test_docs` folder (other formats not supported). Stop with `docker-compose down`. All services expose REST APIs on ports 8000-8005 for programmatic access.

---

## The "Why" - Traditional RAG vs. GraphRAG

Traditional RAG systems embed document chunks and retrieve similar ones via vector search. This works for direct questions but fails when queries require:

1. **Multi-hop reasoning**: Connecting facts across multiple documents
2. **Global understanding**: "What are the main themes across all documents?"
3. **Implicit relationships**: Entities mentioned separately with no lexical overlap

**Example:** *"How do European regulations indirectly impact Asian semiconductor supply chains?"*

- **Traditional RAG**: Retrieves separate chunks about Europe and Asia, misses the connection chain (EU regulation → German manufacturer → Taiwanese supplier → Japanese chipmaker)
- **GraphRAG**: Builds knowledge graph linking entities across documents, detects community clusters, traces multi-hop relationships to provide comprehensive answers

**Key advantage**: GraphRAG enables "forest-view" reasoning (global patterns) alongside "tree-view" retrieval (specific facts).

---

## Solution Architecture & Implementation

### High-Level System Design

![Architecture Diagram](./docs/sol-arch.svg)

The architecture consists of four layers:

- **Client Layer**: Query application submits requests via REST API
- **API Gateway**: Orchestrator Service (Port 8000) coordinates all operations
- **Processing Services**: Document Processor (8004), Graph Processor (8005), LLM Service (8003)
- **Infrastructure Services**: Cache Service (8001) interfaces with Neo4j, Rate Limiter (8002) manages token buckets via Redis
- **Data Layer**: Neo4j (graph storage), Redis (rate limiting state), Fluentd (centralized logging)

### The Funnel: From System Design to Code

The system implements a **distributed microservices architecture** with three tiers:

1. **API Tier**: Orchestrator exposes REST endpoints for indexing and querying
2. **Service Tier**: Specialized microservices handle document processing, graph operations, and LLM interactions
3. **Data Tier**: Neo4j (graph storage), Redis (rate limiting), Fluentd (logging)

**Key Design Decisions:**
- **Stateless services**: Any service replica can handle any request (enables horizontal scaling)
- **Smart caching**: Neo4j stores intermediate pipeline stages (chunks, graphs, community summaries, final answers)
- **Rate limiting**: Centralized Redis-based token bucket prevents LLM quota exhaustion
- **Containerization**: Docker Compose for local dev, easily converted to Kubernetes for production

### Mapping to the "From Local to Global" Paper

| Paper Stage | Implementation Component | Code Location |
|------------|--------------------------|---------------|
| **1. Source Documents** | File loading | `distributed_orchestrator.py:load_documents()` |
| **2. Text Chunks** | Chunking with overlap | `document_service.py:chunk_document()` |
| **3. Element Instances** | LLM extraction | `llm_service.py:extract_elements()` |
| **4. Element Summaries** | LLM summarization | `llm_service.py:summarize_elements()` |
| **5. Graph Communities** | Leiden clustering | `graph_service.py:detect_communities()` |
| **6. Community Summaries** | High-level descriptions | `llm_service.py:summarize_communities()` |
| **7. Community Answers** | Map-reduce query answering | `llm_service.py:answer_query()` + `combine_answers()` |
| **8. Global Answer** | Final response synthesis | `distributed_orchestrator.py:run_pipeline_async()` |

**Key Differences from Paper:**
- **Production caching**: Neo4j persists all intermediate stages (paper assumes fresh computation)
- **Rate limiting**: Token bucket algorithm prevents API quota exhaustion
- **Distributed execution**: Microservices architecture enables horizontal scaling (paper is conceptual)

### Hosting & Infrastructure

This architecture is designed for **flexible deployment** with full control over your infrastructure:

- **On-Premise**: Deploy via Docker Compose on physical servers with Neo4j/Redis on internal infrastructure
- **Cloud (AWS/Azure/GCP)**: Run on ECS/EKS, AKS, or GKE with managed databases (Neo4j Aura, ElastiCache)
- **Hybrid**: Mix cloud orchestration with on-prem data storage for compliance

**LLM Flexibility:** The architecture supports any LLM provider—swap OpenAI for Anthropic Claude, Azure OpenAI, AWS Bedrock, or self-hosted models (Llama via vLLM, Mixtral via TGI) through simple configuration changes. This ensures you're never locked into a single vendor and can optimize for cost, latency, or data sovereignty requirements.

---

## Recommendations - Infrastructure & DevOps

### 1. Managed Database Services with Automated Backups

Migrate Neo4j to managed services like Neo4j Aura or Neo4j Enterprise with automated backups, read replicas, and clustering for high availability. Use AWS ElastiCache, Azure Cache for Redis, or Redis Enterprise for Redis with multi-AZ replication and automatic failover to eliminate single points of failure.

### 2. Infrastructure as Code (OpenTofu/Terraform)

Codify infrastructure using OpenTofu or Pulumi for version-controlled, reproducible deployments. Define all services, databases, networking, and auto-scaling rules in declarative configuration files enabling automated provisioning and consistent dev/staging/prod environments.

### 3. CI/CD Pipelines with Automated Testing

Implement GitHub Actions or GitLab CI workflows for automated testing (unit, integration, security scanning), containerized builds, and zero-downtime blue-green deployments to production environments.

### 4. Observability Stack (Metrics, Traces, Alerts)

Deploy Prometheus + Grafana for metrics dashboards, Jaeger for distributed tracing across services, and PagerDuty/Opsgenie for alerting on critical issues like database connection exhaustion or LLM service errors.

### 5. Kubernetes Migration for Auto-Scaling

Migrate from Docker Compose to Kubernetes (EKS/AKS/GKE) for horizontal pod autoscaling based on CPU/memory, dynamic node provisioning during peak load, and centralized configuration management via ConfigMaps/Secrets.

### 6. Data Retention & Archive Policies

Implement tiered storage with hot data (90 days) in Neo4j, warm data (90 days - 2 years) in S3/Glacier for compliance, and automated purge jobs to prevent unbounded database growth.

---

## Recommendations - AI & Research Methodology

### 1. Deferred "Lazy" Summarization (LazyGraphRAG)

**[LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)** (Microsoft Research, Nov 2024) — Only summarize communities relevant to each query using iterative deepening instead of pre-generating all summaries during indexing. Achieves 60-80% cost reduction while maintaining answer quality.

### 2. Dual-Level Retrieval (LightRAG)

**[LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779)** (Guo et al., Oct 2024) — Implement both low-level entity-to-chunk indexing for factoid queries and high-level community summaries for analytical queries. Route queries to appropriate index based on question type for 3-5x faster responses on specific questions.

### 3. Single-Step Multi-Hop Reasoning (HippoRAG)

**[HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)** (Gutiérrez et al., May 2024) — Use Personalized PageRank to extract relevant subgraphs from query entities, identifying hidden multi-hop connections instantly rather than relying on community summaries to capture them. Improves multi-hop reasoning accuracy by 20-30%.

### 4. Optimized Hybridization (HybridRAG)

**[HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction](https://arxiv.org/abs/2408.04948)** (Mehta et al., Aug 2024) — Run graph-based and vector-based retrieval in parallel, then fuse results for final answer. Catches facts missed by entity extraction and improves recall by 15-25%, especially for technical documents with dense jargon.

---

## Other Recommendations

**Document Cache Invalidation:** Implement file watchers or content-hash tracking to automatically refresh cached graphs when source documents are modified, ensuring query results reflect the latest content.

**Performance Optimization:** Profile LLM call patterns to identify batching opportunities, tune Neo4j query indexes for frequently accessed graph patterns, etc.

---

## Conclusion

This reference architecture demonstrates how to build production-grade GraphRAG systems with full architectural transparency and deployment flexibility. Enterprises should adapt this blueprint to their specific security, compliance, and performance requirements.

---

## License

This project is licensed under the [LICENSE](LICENSE) file in the repository.

## Acknowledgments

- **Microsoft Research**: For the foundational "From Local to Global" GraphRAG paper
- **Open-source community**: Neo4j, Redis, FastAPI, igraph, and all other dependencies
