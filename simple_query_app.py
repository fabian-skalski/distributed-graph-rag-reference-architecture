#!/usr/bin/env python3
"""Simple interactive terminal app for Graph RAG querying."""
import os
import asyncio
import httpx

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header():
    """Print welcome header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print("   Graph RAG Query System")
    print(f"{'='*60}{Colors.ENDC}\n")


def print_menu():
    """Print main menu."""
    print(f"{Colors.OKCYAN}Please select an option:{Colors.ENDC}")
    print(f"  {Colors.BOLD}1{Colors.ENDC} - Index documents")
    print(f"  {Colors.BOLD}2{Colors.ENDC} - Query documents")
    print(f"  {Colors.BOLD}3{Colors.ENDC} - Clear documents cache")
    print(f"  {Colors.BOLD}4{Colors.ENDC} - Exit")
    print()


async def index_documents(client: httpx.AsyncClient, orchestrator_url: str):
    """Index documents."""
    folder = input(f"{Colors.OKBLUE}Enter folder name (default: test_docs): {Colors.ENDC}").strip()
    if not folder:
        folder = "test_docs"
    
    print(f"\n{Colors.OKCYAN}Starting indexing of '{folder}'...{Colors.ENDC}")
    
    try:
        response = await client.post(
            f"{orchestrator_url}/api/index",
            json={"documents_folder": folder},
            timeout=300.0
        )
        response.raise_for_status()
        result = response.json()
        job_id = result.get("job_id")
        
        print(f"{Colors.OKGREEN}✓ Indexing started (Job ID: {job_id}){Colors.ENDC}")
        print(f"{Colors.OKCYAN}Waiting for indexing to complete...{Colors.ENDC}\n")
        
        # Poll for completion
        for _ in range(60):
            await asyncio.sleep(2)
            status_response = await client.get(f"{orchestrator_url}/api/status/{job_id}")
            status = status_response.json()
            
            if status["status"] == "completed":
                docs = status.get("documents_processed", "unknown")
                print(f"{Colors.OKGREEN}✓ Indexing completed! Processed {docs} documents.{Colors.ENDC}\n")
                return
            elif status["status"] == "failed":
                print(f"{Colors.FAIL}✗ Indexing failed: {status.get('error')}{Colors.ENDC}\n")
                return
        
        print(f"{Colors.WARNING}⚠ Indexing taking longer than expected...{Colors.ENDC}\n")
    
    except Exception as e:
        print(f"{Colors.FAIL}✗ Error: {e}{Colors.ENDC}\n")


async def query_documents(client: httpx.AsyncClient, orchestrator_url: str):
    """Query documents."""
    query = input(f"{Colors.OKBLUE}Enter your question: {Colors.ENDC}").strip()
    
    if not query:
        print(f"{Colors.WARNING}⚠ No query entered.{Colors.ENDC}\n")
        return
    
    print(f"\n{Colors.OKCYAN}Processing query...{Colors.ENDC}\n")
    
    try:
        response = await client.post(
            f"{orchestrator_url}/api/query",
            json={"query": query, "documents_folder": "test_docs"},
            timeout=300.0
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"{Colors.BOLD}{Colors.HEADER}Question:{Colors.ENDC} {query}")
        print(f"{Colors.BOLD}{Colors.OKGREEN}Answer:{Colors.ENDC}")
        print(f"{result['answer']}\n")
    
    except Exception as e:
        print(f"{Colors.FAIL}✗ Error: {e}{Colors.ENDC}\n")


async def clear_cache(client: httpx.AsyncClient, orchestrator_url: str):
    """Clear all Neo4j cache."""
    confirm = input(f"{Colors.WARNING}⚠ This will delete ALL data from Neo4j. Continue? (yes/no): {Colors.ENDC}").strip().lower()
    
    if confirm != "yes":
        print(f"{Colors.OKCYAN}Cache clear cancelled.{Colors.ENDC}\n")
        return
    
    print(f"\n{Colors.OKCYAN}Clearing Neo4j cache...{Colors.ENDC}")
    
    try:
        # Get the cache service URL from environment or use default
        cache_url = os.getenv("CACHE_SERVICE_URL", "http://localhost:8001")
        
        # Clear all data
        response = await client.delete(f"{cache_url}/admin/clear-all", timeout=30.0)
        response.raise_for_status()
        result = response.json()
        
        print(f"{Colors.OKGREEN}✓ {result['message']}{Colors.ENDC}\n")
    
    except Exception as e:
        print(f"{Colors.FAIL}✗ Error clearing cache: {e}{Colors.ENDC}\n")



async def main():
    """Main application loop."""
    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
    
    print_header()
    
    # Check connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{orchestrator_url}/health", timeout=5.0)
            response.raise_for_status()
            print(f"{Colors.OKGREEN}✓ Connected to orchestrator{Colors.ENDC}\n")
    except Exception as e:
        print(f"{Colors.FAIL}✗ Cannot connect to orchestrator at {orchestrator_url}{Colors.ENDC}")
        print(f"{Colors.FAIL}  Error: {e}{Colors.ENDC}\n")
        return
    
    # Main loop
    async with httpx.AsyncClient() as client:
        while True:
            print_menu()
            choice = input(f"{Colors.BOLD}Enter choice (1-4): {Colors.ENDC}").strip()
            print()
            
            if choice == "1":
                await index_documents(client, orchestrator_url)
            elif choice == "2":
                await query_documents(client, orchestrator_url)
            elif choice == "3":
                await clear_cache(client, orchestrator_url)
            elif choice == "4":
                print(f"{Colors.OKCYAN}Goodbye!{Colors.ENDC}\n")
                break
            else:
                print(f"{Colors.WARNING}Invalid choice. Please enter 1, 2, 3, or 4.{Colors.ENDC}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.OKCYAN}Interrupted. Goodbye!{Colors.ENDC}\n")
