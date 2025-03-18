import chromadb
import argparse
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import os

def get_chroma_client():
    persist_directory = "/Users/charlesdecian/Documents/RAG/RAG/chroma_db"
    if not os.path.exists(persist_directory):
        print(f"[red]Error: ChromaDB directory not found at {persist_directory}[/red]")
        return None
    return chromadb.PersistentClient(path=persist_directory)

def list_collections(client):
    """List all collections in the ChromaDB instance."""
    collection_names = client.list_collections()
    console = Console()
    
    if not collection_names:
        console.print("[yellow]No collections found in ChromaDB.[/yellow]")
        return
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Collection Name")
    
    for name in collection_names:
        # In v0.6.0, we get just the name
        table.add_row(name)
    
    console.print("\n[bold]Available Collections:[/bold]")
    console.print(table)
    console.print("\nTo view contents of a collection, use:")
    console.print("[blue]python cli.py --collection <collection_name>[/blue]")

def show_collection_contents(client, collection_name):
    """Show contents of a specific collection."""
    try:
        collection = client.get_collection(collection_name)
        result = collection.get()
        
        console = Console()
        
        if not result['ids']:
            console.print("[yellow]No documents found in collection.[/yellow]")
            return
        
        # Document Overview
        doc_table = Table(show_header=True, header_style="bold blue")
        doc_table.add_column("ID")
        doc_table.add_column("Metadata")
        doc_table.add_column("Content Preview")
        
        for idx, doc_id in enumerate(result['ids']):
            metadata = result['metadatas'][idx] if result['metadatas'] else {}
            content = result['documents'][idx] if result['documents'] else "No content"
            content_preview = content[:100] + "..." if len(content) > 100 else content
            
            doc_table.add_row(
                doc_id,
                json.dumps(metadata, indent=2),
                content_preview
            )
        
        console.print("\n[bold]Document Contents:[/bold]")
        console.print(doc_table)
        
        # Statistics
        console.print("\n[bold]Collection Statistics:[/bold]")
        stats = {
            "Total Documents": len(result['ids']),
            "Unique Sources": len(set(m.get('source', '') for m in result['metadatas'])) if result['metadatas'] else 0
        }
        
        stats_table = Table(show_header=True, header_style="bold green")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        
        for metric, value in stats.items():
            stats_table.add_row(metric, str(value))
        
        console.print(stats_table)
    except Exception as e:
        console = Console()
        console.print(f"[red]Error accessing collection: {str(e)}[/red]")

def main():
    parser = argparse.ArgumentParser(description='ChromaDB Inspector')
    parser.add_argument('--list', action='store_true', help='List all collections')
    parser.add_argument('--collection', type=str, help='Show contents of specific collection')
    
    args = parser.parse_args()
    
    client = get_chroma_client()
    if not client:
        return
    
    if args.list:
        list_collections(client)
    elif args.collection:
        show_collection_contents(client, args.collection)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 