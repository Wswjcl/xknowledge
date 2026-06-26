"""QAWiki CLI - command-line interface for knowledge operations."""

import argparse
import sys
import os

from .config import QAWikiConfig, load_config, print_config
from .knowledge import QAKnowledge
from .governance import Actor, Role, coerce_role, PermissionDenied


def _actor_from_args(args) -> Actor:
    """Build an Actor from the global --as/--role flags."""
    user = getattr(args, "as_user", None) or "cli"
    role = coerce_role(getattr(args, "as_role", None) or "editor")
    return Actor(user_id=user, role=role)


def _kb_with_actor(args) -> QAKnowledge:
    """Construct a QAKnowledge and set its actor from CLI flags."""
    kb = QAKnowledge()
    kb.set_actor(_actor_from_args(args))
    return kb


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

    total_added = 0
    total_removed = 0
    for f in files:
        print(f"Ingesting: {f}")
        res = kb.learn_from_document(f)
        action = res["action"]
        d = res["diff"]
        if action == "unchanged":
            print(f"  Skipped (unchanged).")
            continue
        print(
            f"  [{action}] sections: +{d['added']} ~{d['changed']} "
            f"-{d['removed']} ={d['unchanged']} | "
            f"insights: +{res['added_insights']} -{res['removed_insights']} | "
            f"framework: {'yes' if res['framework_generated'] else 'no'}"
        )
        total_added += res["added_insights"]
        total_removed += res["removed_insights"]

    print(f"\nTotal: +{total_added} / -{total_removed} insights from {len(files)} file(s)")
    print(f"KB status: {kb.status['insight_count']} insights, "
          f"{kb.status['framework_words']} framework words, "
          f"{kb.status['doc_count']} documents registered")


def cmd_sources(args):
    """List registered source documents."""
    kb = QAKnowledge()
    docs = kb.registry.list_docs()
    if not docs:
        print("No documents registered yet.")
        return

    print(f"Registered source documents ({len(docs)}):\n")
    for d in docs:
        secs = d.get("sections", {})
        insight_refs = sum(len(s.get("insight_ids", [])) for s in secs.values())
        print(f"  {d['doc_path']}")
        print(f"    ingested: {d.get('ingested_at', '?')} | "
              f"sections: {len(secs)} | insight refs: {insight_refs}")
    print(f"\nRegistry: {kb.status['registry_path']}")


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


def cmd_add(args):
    """Add a new source document, auto-routing by source type."""
    kb = _kb_with_actor(args)
    path = args.path
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)]
    else:
        files = [path]

    for f in files:
        if not os.path.isfile(f):
            continue
        print(f"Adding: {f}")
        try:
            res = kb.add_source(
                f, declared_type=args.type, title=args.title
            )
        except PermissionDenied as e:
            print(f"  ✗ Permission denied: {e}")
            continue
        except FileNotFoundError as e:
            print(f"  ✗ File not found: {e}")
            continue

        route = res.get("route")
        cls = res.get("classification", {})
        print(f"  route: {route} (source_type={res['source_type']}, method={cls.get('method')})")
        if route == "distilled":
            d = res.get("diff", {})
            print(f"    sections: +{d.get('added',0)} ~{d.get('changed',0)} "
                  f"-{d.get('removed',0)} | insights: +{res.get('added_insights',0)}")
        elif route == "reference":
            print(f"    doc_id: {res.get('doc_id')} | version: {res.get('version')} "
                  f"| state: {res.get('state')}")
    print(f"\nKB status: {kb.status['insight_count']} insights "
          f"({kb.status['published_insights']} published, "
          f"{kb.status['pending_insights']} pending), "
          f"{kb.status['reference_count']} references, "
          f"{kb.status['pending_reviews']} pending reviews")


def cmd_review(args):
    """List pending items in the review queue."""
    kb = _kb_with_actor(args)
    pending = kb.list_pending_reviews()
    if not pending:
        print("No pending items to review. 🎉")
        return
    print(f"Pending review queue ({len(pending)}):\n")
    for i in pending:
        print(f"  [{i['id']}] {i['target_type']:9s} {i['target_id']:20s} "
              f"— {i['title'][:50]}")
        print(f"       by {i['proposed_by']}: {i['reason'][:60]}")
    print(f"\nUse: qawiki approve <id>  or  qawiki reject <id>")


def cmd_approve(args):
    """Approve a pending review item."""
    kb = _kb_with_actor(args)
    item = _resolve_review_item(kb, args.item_id)
    if item is None:
        print(f"Review item {args.item_id!r} not found.")
        sys.exit(1)
    try:
        if item["target_type"] == "insight":
            res = kb.approve_insight(item["target_id"])
        else:
            res = kb.approve_reference(item["target_id"])
        print(f"✓ Approved {item['target_type']} {item['target_id']} "
              f"-> state={res.get('new_state')}")
    except PermissionDenied as e:
        print(f"✗ Permission denied: {e}")
        sys.exit(1)


def cmd_reject(args):
    """Reject a pending review item."""
    kb = _kb_with_actor(args)
    item = _resolve_review_item(kb, args.item_id)
    if item is None:
        print(f"Review item {args.item_id!r} not found.")
        sys.exit(1)
    try:
        if item["target_type"] == "insight":
            res = kb.reject_insight(item["target_id"])
        else:
            res = kb.reject_reference(item["target_id"])
        print(f"✗ Rejected {item['target_type']} {item['target_id']} "
              f"-> state={res.get('new_state')}")
    except PermissionDenied as e:
        print(f"✗ Permission denied: {e}")
        sys.exit(1)


def _resolve_review_item(kb: QAKnowledge, item_id: str):
    """Find a review item by its rv-XXXX id (works on target_id too)."""
    pending = kb.list_pending_reviews()
    for i in pending:
        if i["id"] == item_id or i["target_id"] == item_id:
            return i
    return None


def cmd_status(args):
    """Show knowledge base status."""
    kb = QAKnowledge()
    config = load_config()
    print_config(config)
    print()
    s = kb.status
    print(f"KB Directory: {s['kb_dir']}")
    print(f"Insights:     {s['insight_count']} ({s['published_insights']} published, "
          f"{s['pending_insights']} pending)")
    print(f"Framework:    {s['framework_words']} words")
    print(f"Documents:    {s['doc_count']} registered, {s['reference_count']} references")
    print(f"Pending reviews: {s['pending_reviews']}")
    print(f"Insight Path: {s['insight_path']}")
    print(f"Framework:    {s['framework_path']}")
    print(f"Registry:     {s['registry_path']}")


def main():
    parser = argparse.ArgumentParser(
        prog="qawiki",
        description="QAWiki - Team knowledge base with continual learning",
    )
    # Global identity flags (parsed for every subcommand).
    parser.add_argument("--as", dest="as_user", default="cli",
                        help="Act as this user id (default: cli)")
    parser.add_argument("--role", dest="as_role", default="editor",
                        choices=[r.value for r in Role],
                        help="Act with this role (default: editor)")
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

    # qawiki add (unified source intake with routing)
    p_add = sub.add_parser("add", help="Add a source (auto-routes distill vs reference)")
    p_add.add_argument("path", help="Path to document or directory")
    p_add.add_argument("--type", choices=["distillable", "reference"], default=None,
                       help="Force source type (overrides auto-detection)")
    p_add.add_argument("--title", default=None, help="Title (used for reference docs)")
    p_add.set_defaults(func=cmd_add)

    # qawiki review
    p_review = sub.add_parser("review", help="List pending items awaiting review")
    p_review.set_defaults(func=cmd_review)

    # qawiki approve
    p_approve = sub.add_parser("approve", help="Approve a pending review item")
    p_approve.add_argument("item_id", help="Review item id (rv-XXXX) or target id")
    p_approve.set_defaults(func=cmd_approve)

    # qawiki reject
    p_reject = sub.add_parser("reject", help="Reject a pending review item")
    p_reject.add_argument("item_id", help="Review item id (rv-XXXX) or target id")
    p_reject.set_defaults(func=cmd_reject)

    # qawiki sources
    p_sources = sub.add_parser("sources", help="List registered source documents")
    p_sources.set_defaults(func=cmd_sources)

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
