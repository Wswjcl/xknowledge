"""QAWiki CLI - command-line interface for knowledge operations."""

import argparse
import sys


def cmd_ingest(args):
    """Ingest documents into the knowledge base."""
    print(f"Ingesting: {args.path}")


def cmd_search(args):
    """Search the knowledge base."""
    print(f"Searching: {args.query}")


def cmd_status(args):
    """Show knowledge base status."""
    from qawiki.core import load_insight_library
    print("QAWiki Knowledge Base Status")


def main():
    parser = argparse.ArgumentParser(prog="qawiki", description="QAWiki - Team Knowledge Base")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest documents into knowledge base")
    p_ingest.add_argument("path", help="Path to document or directory")

    p_search = sub.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("query", help="Search query")

    p_status = sub.add_parser("status", help="Show knowledge base status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"ingest": cmd_ingest, "search": cmd_search, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
