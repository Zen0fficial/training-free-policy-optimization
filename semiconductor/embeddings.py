from __future__ import annotations

import os
import json
import glob
import hashlib
import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm import LLM


@dataclass
class ExperienceRecord:
    problem_key: str
    experience_id: str
    text: str


def load_all_experiences(experiences_dir: str) -> list[ExperienceRecord]:
    assert os.path.isdir(experiences_dir), f"Directory not found: {experiences_dir}"
    records: list[ExperienceRecord] = []
    for path in sorted(glob.glob(os.path.join(experiences_dir, "*.json"))):
        problem_key = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for exp_id, text in data.items():
                    txt = (text or "").strip()
                    if not txt:
                        continue
                    records.append(
                        ExperienceRecord(
                            problem_key=problem_key,
                            experience_id=str(exp_id),
                            text=txt,
                        )
                    )
        except Exception:
            continue
    return records


def _normalize_problem(text: str) -> str:
    t = str(text).strip().lower()
    # Collapse whitespace to single spaces for stable hashing
    return " ".join(t.split())


def compute_problem_key(problem: str) -> str:
    norm = _normalize_problem(problem)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def build_prompts_map_from_dataset(data_path: str | None) -> dict[str, str]:
    """Build {problem_key: prompt} from a dataset of QA dicts.

    Accepts JSON (array of dicts) or JSONL. For each row, it uses the first
    available field among ['problem', 'question', 'prompt'] as the prompt text.
    """
    if not data_path:
        return {}
    if not os.path.exists(data_path):
        return {}
    mapping: dict[str, str] = {}
    try:
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    p = row.get("problem") or row.get("question") or row.get("prompt")
                    if not p:
                        continue
                    prompt = str(p).strip()
                    if not prompt:
                        continue
                    key = compute_problem_key(prompt)
                    mapping[key] = prompt
        else:
            # Assume JSON array of dicts
            data = json.load(open(data_path, "r", encoding="utf-8"))
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    p = row.get("problem") or row.get("question") or row.get("prompt")
                    if not p:
                        continue
                    prompt = str(p).strip()
                    if not prompt:
                        continue
                    key = compute_problem_key(prompt)
                    mapping[key] = prompt
    except Exception:
        return {}
    return mapping


def encode_texts(model: LLM, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    all_vecs: list[list[float]] = []
    step = max(1, batch_size)
    for i in range(0, len(texts), step):
        outputs = model.embed(texts[i : i + step])
        all_vecs.extend([o.outputs.embedding for o in outputs])
    arr = np.asarray(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _sha1(text: str) -> str:
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _save_outputs(index_dir: str, embeddings: np.ndarray, prompt_infos: list[dict[str, str]], model_id: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(index_dir, "prompt_sha1.json"), "w", encoding="utf-8") as f:
        json.dump(prompt_infos, f, ensure_ascii=False)
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"model_id": model_id, "embedding_dim": int(embeddings.shape[1]), "num_rows": int(embeddings.shape[0])},
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_texts_and_prompt_shas(
    experiences_dir: str,
    prompts_map: dict[str, str],
) -> tuple[list[str], list[dict[str, str]], list[ExperienceRecord]]:
    """Create one combined text per prompt in a deterministic order.

    Order: iterate prompts_map keys in dataset insertion order; for each problem,
    concatenate all non-empty experience texts (sorted by experience_id) after
    the prompt so that each prompt yields exactly one embedding row.
    Returns (texts_per_prompt, prompt_info_per_prompt, records_in_row_order).
    Each prompt_info row contains {"problem_key": key, "prompt": prompt} aligned to embeddings.
    """
    texts: list[str] = []
    prompt_infos: list[dict[str, str]] = []
    row_records: list[ExperienceRecord] = []
    # Preserve insertion order of prompts_map (Python 3.7+ dicts are ordered)
    for problem_key, prompt in prompts_map.items():
        exp_path = os.path.join(experiences_dir, f"{problem_key}.json")
        if not os.path.exists(exp_path):
            # Assume empty experiences for this problem_key
            continue
        try:
            with open(exp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Malformed file: treat as no experiences
            continue
        if not isinstance(data, dict) or not data:
            # No experiences: skip this prompt
            continue
        exp_texts: list[str] = []
        for exp_id in sorted(map(str, data.keys())):
            exp_text = str(data.get(exp_id, "")).strip()
            if exp_text:
                exp_texts.append(exp_text)
        if not exp_texts:
            continue
        combined_text = f"{prompt} " + " ".join(exp_texts)
        texts.append(combined_text)
        prompt_infos.append({"problem_key": problem_key, "prompt": prompt})
        # Track synthetic row aligned to this combined embedding row
        row_records.append(
            ExperienceRecord(
                problem_key=problem_key,
                experience_id="*",
                text=combined_text,
            )
        )
    return texts, prompt_infos, row_records


# No search API here; this module only builds and saves embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract embeddings in a strict pipeline: load prompts -> hash -> retrieve experiences -> embed -> save."
        )
    )
    parser.add_argument("--data-file", required=True, help="Dataset JSON/JSONL (list of QA dicts) to build prompt map")
    parser.add_argument("--experiences-dir", required=True, help="Directory of per-problem experiences JSON files")
    parser.add_argument("--index-dir", required=True, help="Output directory for embeddings and prompt hashes")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-8B", help="Embedding model id for vLLM")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for embedding texts (0=process all in one batch)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM embedding engine")
    parser.add_argument("--dtype", default="auto", help="dtype for embedding engine (e.g., auto, float16, bfloat16)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.1, help="GPU memory fraction for embedding engine (e.g., 0.1)")

    args = parser.parse_args()

    # 1) Load prompts and compute problem keys (ordered)
    prompts_map = build_prompts_map_from_dataset(args.data_file)
    if not prompts_map:
        raise ValueError("No prompts loaded from dataset; cannot proceed.")

    # 2) Retrieve experiences and build combined texts + prompt hashes in row order
    texts, prompt_infos, _ = build_texts_and_prompt_shas(args.experiences_dir, prompts_map)

    # 3) Build vLLM engine directly and embed in batches
    model = LLM(
        model=args.model_id,
        task="embed",
        trust_remote_code=True,
        tensor_parallel_size=int(args.tensor_parallel_size),
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
    )
    eff_bs = args.batch_size if args.batch_size and args.batch_size > 0 else len(texts)
    emb = encode_texts(model, texts, batch_size=eff_bs)

    # 4) Save outputs aligned by row
    _save_outputs(args.index_dir, emb, prompt_infos, args.model_id)

    print(json.dumps({
        "num_rows": len(texts),
        "embedding_dim": int(emb.shape[1]) if emb.size > 0 else 0,
        "index_dir": args.index_dir,
        "model_id": args.model_id,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "data_file": args.data_file,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


