from __future__ import annotations

import json
import os
import random
import re
import hashlib
import glob
from typing import Callable
from dataclasses import dataclass

import pandas as pd
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import LLM


# -------------------------
# Data loading & verification
# -------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_batch_prompts(path: str) -> list[str]:
    assert os.path.exists(path), f"Input not found: {path}"
    if path.endswith('.json'):
        prompts: list[str] = []
        data = load_json(path)
        for dt in data:
            p = dt.get('prompt') or dt.get('problem') or dt.get('question')
            if p:
                prompts.append(str(p))
        return prompts
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_local_dataset(path: str) -> list[dict]:
    assert os.path.exists(path), f"Dataset not found: {path}"
    if path.endswith(".jsonl"):
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
        data = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {path}")

    def _parse_keypoints(val) -> list[str] | None:
        """Parse and validate keypoints in required format.

        Accepted inputs:
        - str: newline-separated lines, first is summary, following are numbered lines '1. ', '2. ', ...
        - str: newline-separated lines with numbering '1. ', '2. ', ...
        - list[str]: first element is summary, following elements are numbered '1. ', '2. ', ...

        Returns list[str] as [summary, '1. ...', '2. ...', ...] when valid. None if invalid.
        """
        lines: list[str] = []
        if isinstance(val, str):
            lines = [ln.strip() for ln in val.splitlines() if ln.strip()]
        elif isinstance(val, list):
            lines = [str(x).strip() for x in val if str(x).strip()]
        else:
            return None
        summary = lines[0]
        points = lines[1:]
        # Validate strict numbering 1..N
        if re.match(rf"^1\.\s+", summary):
            for i, p in enumerate(points, start=1):
                if not re.match(rf"^{i+1}\.\s+", p):
                    return None
            return [summary] + points
        else:
            for i, p in enumerate(points, start=1):
                if not re.match(rf"^{i}\.\s+", p):
                    return None
            return [f"0. {summary}"] + points

    normalized = []
    for i, row in enumerate(data):
        problem = row.get("problem") or row.get("question") or row.get("prompt")
        groundtruth = row.get("groundtruth") or row.get("answer") or row.get("response")
        if problem is None or groundtruth is None:
            raise ValueError("Each row must include 'problem' and 'groundtruth' (or 'question'/'answer').")
        keypoints_parsed = _parse_keypoints(row.get("keypoints") or row.get("point"))
        if keypoints_parsed is None:
            # Drop rows that do not obey the required keypoints format
            print(f"Dropping row {i}: {row}")
            continue
        normalized.append(
            {
                "problem": problem,
                "groundtruth": groundtruth,
                "keypoints": keypoints_parsed,
                **{k: v for k, v in row.items() if k not in ["problem", "question", "prompt", "groundtruth", "answer", "keypoints"]},
            }
        )
    return normalized


# -------------------------
# Per-problem experience storage helpers
# -------------------------

def _normalize_problem(text: str) -> str:
    t = str(text).strip().lower()
    # Collapse whitespace to single spaces to make hashing stable
    t = re.sub(r"\s+", " ", t)
    return t


def compute_problem_key(problem: str) -> str:
    """Compute a short, stable key for a problem string.

    Uses SHA1 over a normalized version of the text and returns the first 12 hex chars.
    """
    norm = _normalize_problem(problem)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def load_experiences_for_prompt(problem: str, root_dir: str) -> dict[str, str]:
    """Load per-problem experiences given a problem string.

    - `root_dir` should be a directory like ".../experiences_by_problem".
    Returns an ID->text mapping or an empty dict when none exist.
    """
    # Use the provided root_dir strictly; no environment overrides.
    key = compute_problem_key(problem)
    path = os.path.join(root_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Ensure the shape is a dict[str, str]
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass
    return {}


def save_experiences_for_problem(problem: str, experiences: dict[str, str], root_dir: str) -> None:
    """Persist per-problem experiences to `<root_dir>/<key>.json`."""
    os.makedirs(root_dir, exist_ok=True)
    key = compute_problem_key(problem)
    path = os.path.join(root_dir, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(experiences or {}, f, indent=2, ensure_ascii=False)


# -------------------------
# Grading & JSON helpers
# -------------------------
def format_requirements_block(reqs: list[str]) -> str:
    """Format keypoints into required grading block.

    Expected input: list where the first item is the summary, followed by key points.
    Output format:
      0. <summary>
      1. <point 1>
      2. <point 2>
      ...
    If points already numbered 1..N in order, preserve their text and only prepend '0. ' to the summary.
    Otherwise, re-number points to enforce 1..N.
    """
    if not reqs:
        return "无"

    summary = str(reqs[0]).strip()
    raw_points = [str(x).strip() for x in reqs[1:]]

    # Check existing numbering 1..N
    valid = True
    for i, p in enumerate(raw_points, start=1):
        if not re.match(rf"^{i}\.\s+", p):
            valid = False
            break

    if not valid:
        # Invalid numbering; return empty to signal failure
        return ""

    block_lines = [f"0. {summary}"] + raw_points
    return "\n".join(block_lines)


def safe_json_obj(text: str) -> dict:
    try:
        if "```" in text:
            inner = text.split("```json")[-1].split("```")[0]
            data = json.loads(inner)
            return data if isinstance(data, dict) else {}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def safe_json_array(text: str) -> list:
    try:
        if "```" in text:
            inner = text.split("```json")[-1].split("```")[0]
            data = json.loads(inner)
            return data if isinstance(data, list) else []
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, list) else []
    except Exception:
        return []
    return []


def extract_final_answer(response):
    """提取最终答案（移除思考过程）
    
    如果回答包含</think>标签，提取标签后的内容作为最终答案
    否则返回原始文本（去除首尾空白）
    """
    if "</think>" in response:
        response_parts = response.split("</think>")
        return response_parts[1].strip()  # 返回标签后的内容
    return str(response).strip()


def to_qwen_thinking_chat(
    user_text: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str:
    """Format text using Qwen3 chat template with enable_thinking=True.
    """
    text = str(user_text)
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


# -------------------------
# Embeddings & retrieval utilities (shared)
# -------------------------


@dataclass
class ExperienceRecord:
    problem_key: str
    experience_id: str
    text: str
    source_file: str


def load_all_experiences(experiences_dir: str) -> list[ExperienceRecord]:
    """Load all experiences from a per-problem directory.

    Each file is expected to be `<problem_key>.json` with a JSON object mapping
    experience IDs to text.
    """
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
                            source_file=path,
                        )
                    )
        except Exception:
            # Skip malformed files but continue scanning others
            continue
    return records


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


def cosine_topk(query_vec: np.ndarray, doc_matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return top-k indices with scores from a normalized matrix using cosine similarity."""
    # Defensive handling: empty matrix or vector
    if doc_matrix is None or doc_matrix.size == 0 or doc_matrix.shape[0] == 0:
        return []
    if query_vec is None or query_vec.size == 0:
        return []
    # Dimension check for clear error messages
    if doc_matrix.shape[1] != query_vec.reshape(-1).shape[0]:
        raise ValueError(
            f"cosine_topk dimension mismatch: doc_matrix shape={doc_matrix.shape}, "
            f"query_vec shape={query_vec.shape}. This often indicates the index "
            f"was built with a different embedding model or there are no documents."
        )
    scores = (doc_matrix @ query_vec.reshape(-1, 1)).reshape(-1)
    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def apply_instruction(texts: list[str], instruction: str | None) -> list[str]:
    if not instruction:
        return texts
    prefix = instruction.strip()
    return [f"{prefix} {t}" for t in texts]


def _sha1(text: str) -> str:
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _record_uid(r: ExperienceRecord, instruction: str | None) -> str:
    """Stable UID for caching embeddings; includes instruction since it affects vectors."""
    parts = [r.problem_key, r.experience_id, r.text, instruction or ""]
    return _sha1("|".join(parts))[:16]


def _load_existing_index(index_dir: str) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    meta_path = os.path.join(index_dir, "items.json")
    emb_path = os.path.join(index_dir, "embeddings.npy")
    if not (os.path.exists(meta_path) and os.path.exists(emb_path)):
        return None, None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        emb = np.load(emb_path)
        if not isinstance(items, list):
            return None, None
        return items, emb
    except Exception:
        return None, None


def _save_index(index_dir: str, items: list[dict], embeddings: np.ndarray, metadata: dict) -> None:
    os.makedirs(index_dir, exist_ok=True)
    meta_path = os.path.join(index_dir, "items.json")
    emb_path = os.path.join(index_dir, "embeddings.npy")
    header_path = os.path.join(index_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    np.save(emb_path, embeddings)
    with open(header_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def build_or_load_index(
    experiences_dir: str,
    index_dir: str,
    model: LLM,
    batch_size: int,
    instruction: str | None,
) -> tuple[list[ExperienceRecord], np.ndarray]:
    """Create an on-disk embedding index for experience texts (always rebuild).

    Purpose
    - Maintain a persistent, row-aligned embedding matrix for all experiences stored
      under `experiences_dir`, along with a metadata list describing each row.

    Inputs
    - experiences_dir: Directory containing per-problem JSON files. Each file is
      `<problem_key>.json` with a JSON object mapping `experience_id -> text`.
    - index_dir: Directory to cache the built index. Files written:
      - `items.json`: list of dicts aligned to rows with fields:
        {uid, problem_key, experience_id, source_file, text_sha1}
      - `embeddings.npy`: float32 matrix of L2-normalized embeddings (shape N×D)
      - `metadata.json`: {instruction, embedding_dim}
    - model: vLLM engine configured for embedding (`task=embed`).
    - batch_size: Batch size for embedding calls.
    - instruction: Optional prefix applied via `apply_instruction()` before
      embedding. This affects both the vectors and the stable UID.

    Outputs
    - (records, embeddings):
      - records: list[ExperienceRecord], one per row, in stable order.
      - embeddings: np.ndarray of shape (len(records), emb_dim), L2-normalized,
        row-aligned to `records[i]`.

    Logic overview (simplified)
    1) Load all experiences from `experiences_dir` into `records`.
    2) Compute a stable UID per record using `_record_uid(record, instruction)`.
    3) Embed all texts (with instruction applied) and save index files.

    Invariants
    - Embeddings are L2-normalized (see `encode_texts`).
    - Row i in `embeddings` corresponds to `records[i]`.
    - The optional `instruction` consistently affects both document and query
      vectors (when used by retrieval functions).
    """
    records = load_all_experiences(experiences_dir)
    current_uids = [_record_uid(r, instruction) for r in records]

    # Always rebuild the embeddings from current records
    doc_texts = [r.text for r in records]
    doc_texts_for_embed = apply_instruction(doc_texts, instruction)
    emb = encode_texts(model, doc_texts_for_embed, batch_size=batch_size)

    items = [
        {
            "uid": uid,
            "problem_key": r.problem_key,
            "experience_id": r.experience_id,
            "source_file": r.source_file,
            "text_sha1": _sha1(r.text),
        }
        for uid, r in zip(current_uids, records)
    ]
    if index_dir:
        _save_index(
            index_dir,
            items,
            emb,
            {"instruction": instruction or "", "embedding_dim": int(emb.shape[1])},
        )
    return records, emb


def run_search(
    experiences_dir: str,
    query: str | list[str],
    top_k: int,
    device: str | None,
    batch_size: int,
    use_flash_attention_2: bool,
    instruction: str | None,
    index_dir: str | None,
    rebuild_index: bool,
    engine: LLM | None = None,
    exclude_self: bool = True,
) -> list[dict] | list[list[dict]]:
    # Use the provided experiences_dir strictly; no environment overrides.
    if index_dir:
        records, doc_emb = build_or_load_index(
            experiences_dir=experiences_dir,
            index_dir=index_dir,
            model=engine,
            batch_size=batch_size,
            instruction=instruction,
        )
    else:
        records = load_all_experiences(experiences_dir)
        if not records:
            return []
        doc_texts = [r.text for r in records]
        doc_texts_for_embed = apply_instruction(doc_texts, instruction)
        doc_emb = encode_texts(engine, doc_texts_for_embed, batch_size=batch_size)

    if isinstance(query, str):
        query_texts = [query]
        single = True
    else:
        query_texts = list(query)
        single = False

    # If no documents/embeddings are available, return empty results shaped by the query input
    if doc_emb is None or doc_emb.size == 0 or doc_emb.shape[0] == 0 or doc_emb.shape[1] == 0:
        return [] if single else [[] for _ in query_texts]

    queries_for_embed = apply_instruction(query_texts, instruction)
    query_embs = encode_texts(engine, queries_for_embed, batch_size=max(1, min(len(queries_for_embed), 64)))

    # Compute problem_key for each query to support excluding self
    query_keys = [compute_problem_key(q) for q in query_texts]

    # Ensure embedding dimensions match between docs and queries for meaningful similarity
    if query_embs.size > 0 and (doc_emb.shape[1] != query_embs.shape[1]):
        raise ValueError(
            f"Embedding dim mismatch between documents and queries: "
            f"doc_emb.shape={doc_emb.shape}, query_embs.shape={query_embs.shape}. "
            f"Rebuild the index with the same embedding engine/instruction used for queries."
        )

    def build_results_for_query(qvec: np.ndarray, exclude_key: str | None) -> list[dict]:
        # Exclude self BEFORE selection by masking the doc matrix
        if exclude_self and exclude_key:
            mask = np.array([r.problem_key != exclude_key for r in records], dtype=bool)
            if not mask.any():
                return []
            doc_sub = doc_emb[mask]
            # Use fast top-k on the filtered sub-matrix
            top = cosine_topk(qvec, doc_sub, top_k)
            rec_sub = [r for r, keep in zip(records, mask) if keep]
            out: list[dict] = []
            for idx, score in top:
                r = rec_sub[int(idx)]
                out.append(
                    {
                        "score": score,
                        "experience_id": r.experience_id,
                        "text": r.text,
                        "problem_key": r.problem_key,
                        "source_file": r.source_file,
                    }
                )
            return out
        # Fallback: no exclusion, use fast top-k
        top = cosine_topk(qvec, doc_emb, top_k)
        out: list[dict] = []
        for idx, score in top:
            r = records[idx]
            out.append(
                {
                    "score": score,
                    "experience_id": r.experience_id,
                    "text": r.text,
                    "problem_key": r.problem_key,
                    "source_file": r.source_file,
                }
            )
        return out

    all_results = [build_results_for_query(query_embs[i], query_keys[i] if exclude_self else None) for i in range(query_embs.shape[0])]
    return all_results[0] if single else all_results


def format_experiences_for_prompt(items: list[dict]) -> str:
    if not items:
        return "None"
    lines = []
    for it in items:
        tag = f"[{it['problem_key']}:{it['experience_id']}]"
        text = (it.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{tag} {text}")
    return "\n".join(lines) if lines else "None"


def read_batch_prompts(path: str) -> list[str]:
    assert os.path.exists(path), f"Input not found: {path}"
    if path.endswith('.jsonl'):
        prompts: list[str] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                p = obj.get('prompt') or obj.get('problem') or obj.get('question')
                if p:
                    prompts.append(str(p))
        return prompts
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]
