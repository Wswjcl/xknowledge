"""QAWiki CLI - command-line interface for knowledge operations."""

import argparse
import sys
import os

from .config import QAWikiConfig, load_config, print_config
from .knowledge import QAKnowledge


def cmd_run(args):
    """Run the full pipeline: process questions with interleaved distillation."""
    from .pipeline import run_pipeline
    config = load_config()
    run_pipeline(args.input, args.output, config)


def cmd_ingest(args):
    """Ingest documents into the knowledge base."""
    kb = QAKnowledge()
    path = args.path
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if f.endswith((".md", ".markdown", ".docx"))]
    else:
        files = [path]

    total = 0
    for f in files:
        print(f"Ingesting: {f}")
        n = kb.learn_from_document(f)
        total += n
        print(f"  Added {n} insights")

    print(f"\nTotal: {total} insights from {len(files)} files")
    print(f"KB status: {kb.status['insight_count']} insights, "
          f"{kb.status['framework_words']} framework words")


def cmd_search(args):
    """Search the knowledge base for a query."""
    kb = QAKnowledge()
    response = kb.ask(args.query)

    print(f"Query: {args.query}\n")

    if response.get("adapted_guide"):
        print("=" * 60)
        print("ADAPTED GUIDE")
        print("=" * 60)
        print(response["adapted_guide"])

    if response.get("insights"):
        print("\n" + "=" * 60)
        print(f"RELATED INSIGHTS ({len(response['insights'])})")
        print("=" * 60)
        for k, v in response["insights"].items():
            print(f"  [{k}] {v}")

    if response.get("retrieval_info", {}).get("decomposition_used"):
        ri = response["retrieval_info"]
        print(f"\nSubtasks used: {len(ri.get('subtasks', []))}")
        for s in ri.get("subtasks", []):
            print(f"  - [{s.get('type', '?')}] {s.get('query', '')}")


def cmd_status(args):
    """Show knowledge base status."""
    kb = QAKnowledge()
    config = load_config()
    print_config(config)
    print()
    print(f"KB Directory: {kb.status['kb_dir']}")
    print(f"Insights:     {kb.status['insight_count']}")
    print(f"Framework:    {kb.status['framework_words']} words")
    print(f"Insight Path: {kb.status['insight_path']}")
    print(f"Framework:    {kb.status['framework_path']}")


def main():
    parser = argparse.ArgumentParser(
        prog="qawiki",
        description="QAWiki - Team knowledge base with continual learning",
    )
    sub = parser.add_subparsers(dest="command")

    # qawiki run
    p_run = sub.add_parser("run", help="Run full pipeline (Q&A + distillation)")
    p_run.add_argument("--input", "-i", required=True, help="Input JSON/JSONL questions file")
    p_run.add_argument("--output", "-o", default=None, help="Output directory for trajectories")
    p_run.set_defaults(func=cmd_run)

    # qawiki ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into knowledge base")
    p_ingest.add_argument("path", help="Path to document or directory")
    p_ingest.set_defaults(func=cmd_ingest)

    # qawiki search
    p_search = sub.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.set_defaults(func=cmd_search)

    # qawiki status
    p_status = sub.add_parser("status", help="Show knowledge base status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
