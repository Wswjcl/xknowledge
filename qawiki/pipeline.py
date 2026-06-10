"""
Orchestration pipeline - Phase I (distillation) and Phase II (retrieval) interleaved.

Pattern adapted from XSkill eval/infer_api.py
Original work by Jiang et al. (ICML 2026, MIT License)
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import QAWikiConfig, load_config, print_config
from .knowledge import QAKnowledge


class Pipeline:
    """Orchestrates the QA + continual learning pipeline."""

    def __init__(self, config: Optional[QAWikiConfig] = None):
        self.config = config or load_config()
        self.kb = QAKnowledge(self.config)
        self._lock = threading.Lock()

    def run(self, input_file: str, output_dir: Optional[str] = None) -> List[Dict]:
        """Main entry point - process questions with interleaved distillation.

        Args:
            input_file: Path to JSON/JSONL file with questions
            output_dir: Directory for trajectory outputs

        Returns:
            List of result dicts
        """
        print_config(self.config)
        print(f"\nKnowledge Base: {self.kb.status['insight_count']} insights, "
              f"{self.kb.status['framework_words']} framework words\n")

        # Load questions
        questions = self._load_questions(input_file)
        if self.config.max_samples and self.config.max_samples < len(questions):
            questions = questions[:self.config.max_samples]
        print(f"Loaded {len(questions)} questions")

        if not output_dir:
            output_dir = str(self.kb.kb_dir / "trajectories")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        batch_buffer: List[Dict] = []
        batch_idx = 0

        for idx, q in enumerate(tqdm(questions, desc="Processing")):
            sample_dir = os.path.join(output_dir, q["id"])
            os.makedirs(sample_dir, exist_ok=True)

            # Phase II: Retrieve knowledge + generate answer
            response = self.kb.ask(q["text"])

            # For each rollout: simulate agent interaction
            rollout_results = []
            traj_paths = []
            for r in range(self.config.rollouts_per_sample):
                r_dir = os.path.join(sample_dir, f"rollout_{r}")
                os.makedirs(r_dir, exist_ok=True)
                traj_path = os.path.join(r_dir, "traj.jsonl")

                # Simulate: knowledge-guided interaction
                traj = self._simulate_rollout(q, response, r)
                with open(traj_path, "w", encoding="utf-8") as f:
                    for turn in traj:
                        f.write(json.dumps(turn, ensure_ascii=False) + "\n")

                traj_paths.append(traj_path)
                rollout_results.append({
                    "rollout_idx": r,
                    "turns": len(traj),
                    "traj_path": traj_path,
                })

            result = {
                "id": q["id"],
                "question": q["text"],
                "insights_retrieved": len(response.get("insights", {})),
                "adapted_guide_length": len(response.get("adapted_guide", "")),
                "rollouts": rollout_results,
                "sample_dir": sample_dir,
            }
            results.append(result)

            # Accumulate for batch distillation
            batch_buffer.append({
                "sample_idx": idx,
                "question_id": q["id"],
                "sample_dir": sample_dir,
                "question": q["text"],
                "ground_truth": q.get("answer", ""),
                "sample_rollout_results": rollout_results,
                "traj_paths": traj_paths,
            })

            # Trigger large batch distillation
            if len(batch_buffer) >= self.config.large_batch_size:
                self._distill_batch(batch_buffer, batch_idx)
                batch_idx += 1
                batch_buffer = []

        # Final batch
        if batch_buffer:
            self._distill_batch(batch_buffer, batch_idx, is_final=True)

        # Save results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "total": len(results),
                "kb_status": self.kb.status,
                "results": results,
            }, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {results_path}")
        print(f"Final KB: {self.kb.status['insight_count']} insights, "
              f"{self.kb.status['framework_words']} framework words")

        return results

    # ---- Internal ----

    def _load_questions(self, path: str) -> List[Dict]:
        """Load questions from JSON or JSONL file."""
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                items = [json.loads(line) for line in f if line.strip()]
            else:
                items = json.load(f)

        result = []
        for i, item in enumerate(items):
            text = item.get("question") or item.get("problem") or item.get("text", "")
            answer = item.get("answer") or item.get("solution") or item.get("ground_truth", "")
            qid = item.get("id") or item.get("doc_id") or item.get("question_id") or f"q{i:04d}"
            result.append({"id": qid, "text": text, "answer": answer, "raw": item})
        return result

    def _simulate_rollout(self, q: Dict, response: Dict, seed: int) -> List[Dict]:
        """Simulate an agent interaction trajectory.

        In production, this would be replaced by actual Agent execution.
        The trajectory format follows XSkill conventions:
        - turn_idx: 0-based turn number
        - reasoning: agent's thinking
        - tool_call: optional tool invocation
        - tool_result: optional tool output
        """
        traj = [{
            "initial_prompt": q["text"],
            "rollout_seed": seed,
            "injected_knowledge": response.get("adapted_guide", "")[:500],
        }]

        # Simulate 1-3 reasoning turns
        turns = [
            f"Analyzing question: {q['text'][:80]}...",
            "Using retrieved knowledge to formulate answer...",
            "Finalizing response...",
        ]
        for i, reasoning in enumerate(turns):
            traj.append({
                "turn_idx": i,
                "reasoning": reasoning,
                "knowledge_applied": list(response.get("insights", {}).keys())[:3],
                "tool_call": None,
            })

        if q.get("answer"):
            traj.append({
                "turn_idx": len(turns),
                "reasoning": f"Answer: {q['answer']}",
                "ground_truth": q["answer"],
            })

        return traj

    def _distill_batch(self, batch: List[Dict], batch_idx: int, is_final: bool = False):
        """Distill knowledge from a batch of trajectories."""
        start = time.time()
        print(f"\n[Batch {batch_idx}] Distilling from {len(batch)} samples...")

        all_ops = []
        all_frameworks = []

        # Parallel: summarize + critique each sample
        with ThreadPoolExecutor(max_workers=min(self.config.num_workers, len(batch))) as ex:
            futures = {
                ex.submit(self._distill_sample, info): info["question_id"]
                for info in batch
            }
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    ops, fw = future.result()
                    all_ops.extend(ops)
                    if fw:
                        all_frameworks.append(fw)
                except Exception as e:
                    print(f"  Warning: distill failed for {qid}: {e}")

        # Serial: merge
        if all_ops:
            with self._lock:
                prev_count = len(self.kb._insights)
                self.kb._insights = batch_merge_insights(
                    self.kb._insights, all_ops, self.kb.llm,
                    experience_limit=self.config.insight_max_items,
                    similarity_threshold=self.config.insight_similarity_threshold,
                )

                # Refine if over limit
                if len(self.kb._insights) > self.config.insight_max_items or is_final:
                    self.kb._insights = refine_insight_library(
                        self.kb._insights, self.kb.llm,
                    )

                self.kb._save()
                self.kb.retriever.update_experiences(self.kb._insights)
                new_count = len(self.kb._insights)
                print(f"  Insights: {prev_count} -> {new_count} (+{len(all_ops)} ops)")

        if all_frameworks:
            with self._lock:
                self.kb._framework = merge_frameworks(
                    self.kb._framework, all_frameworks, self.kb.llm, None,
                )
                if len(self.kb._framework.split()) > self.config.framework_word_threshold or is_final:
                    self.kb._framework = refine_framework_document(
                        self.kb._framework, self.kb.llm,
                        word_threshold=self.config.framework_word_threshold,
                        force_refine=is_final,
                    )
                self.kb._save()

        elapsed = time.time() - start
        print(f"  [Batch {batch_idx}] Complete ({elapsed:.1f}s)")

    def _distill_sample(self, info: Dict) -> Tuple[List[Dict], Optional[str]]:
        """Distill a single sample: summarize + critique."""
        from .core.summarizer import summarize_rollouts
        from .core.critic import intra_sample_experiences as critique

        traj_paths = info.get("traj_paths", [])
        if not traj_paths:
            # Fallback: find traj files
            sample_dir = info["sample_dir"]
            traj_paths = []
            for r in range(self.config.rollouts_per_sample):
                p = os.path.join(sample_dir, f"rollout_{r}", "traj.jsonl")
                if os.path.exists(p):
                    traj_paths.append(p)

        if not traj_paths:
            return [], None

        # Summarize trajectories
        summary = summarize_rollouts(traj_paths, self.kb.llm, sample_dir=info["sample_dir"])
        if not summary:
            return [], None

        summaries_only = {k: v for k, v in summary.items()
                          if k not in ("question", "ground_truth", "system_prompt")}

        # Cross-critique
        ops = critique(
            info["question"], info.get("ground_truth", ""),
            summaries_only, self.kb.llm,
            max_ops=self.config.insight_max_ops,
            debug_dir=info["sample_dir"],
        )

        norm_ops = []
        for o in (ops if isinstance(ops, list) else []):
            if isinstance(o, dict):
                exp_txt = o.get("experience") or ""
                if exp_txt.strip():
                    norm_ops.append({"experience": exp_txt.strip()})

        # Try framework generation
        fw_content = None
        try:
            from .core.framework import generate_skill_for_sample
            sample_info = {
                "question_id": info["question_id"],
                "sample_dir": info["sample_dir"],
                "sample_rollout_results": info.get("sample_rollout_results", []),
            }
            fw_result = generate_skill_for_sample(
                sample_info, self.kb.llm, None,
                ground_truth=info.get("ground_truth", ""),
            )
            if fw_result.get("success") and fw_result.get("skill_content"):
                fw_content = fw_result["skill_content"]
        except Exception:
            pass

        return norm_ops, fw_content


def run_pipeline(input_file: str, output_dir: Optional[str] = None,
                 config: Optional[QAWikiConfig] = None) -> List[Dict]:
    """Convenience function to run the full pipeline."""
    pipeline = Pipeline(config)
    return pipeline.run(input_file, output_dir)
