from __future__ import annotations

import argparse
import sys
import json
import os
from typing import List
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from training_free_grpo.semiconductor.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
from training_free_grpo.semiconductor.embeddings import (
    build_prompts_map_from_dataset,
    build_texts_and_prompt_shas,
    encode_texts as encode_with_vllm,
)
from training_free_grpo.semiconductor.utils import read_batch_prompts


def _to_qwen_thinking_chat(user_text: str, tokenizer: AutoTokenizer) -> str:
    messages = [{"role": "user", "content": str(user_text)}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_final_answer(response: str) -> str:
    if "</think>" in response:
        parts = response.split("</think>")
        return parts[1].strip()
    return response.strip()



def _format_experiences_for_prompt(rows: list[dict]) -> str:
    if not rows:
        return "None"
    blocks: list[str] = []
    for r in rows:
        pr = (r.get("prompt") or "").strip()
        ex = (r.get("experience") or "").strip()
        if pr and ex:
            blocks.append(f"{pr} {ex}")
        elif ex:
            blocks.append(ex)
        elif pr:
            blocks.append(pr)
    return "\n\n".join(blocks) if blocks else "None"



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate with Qwen3-32B via vLLM while injecting top-k similar experiences from a prebuilt index."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Raw question/prompt text")
    group.add_argument("--query-file", help="File path to read a single prompt from")

    # Required inputs for retrieval
    parser.add_argument("--data-file", required=True, help="Dataset JSON/JSONL to reconstruct prompt order and keys")
    parser.add_argument("--experiences-dir", help="Directory of per-problem experiences JSON files")
    parser.add_argument("--embed-index-dir", required=True, help="Directory containing embeddings.npy and prompt_sha1.json")

    # Retrieval controls
    parser.add_argument("--top-k", type=int, default=5, help="Number of experiences to inject")
    parser.add_argument("--embed-model-id", default="Qwen/Qwen3-Embedding-8B", help="Embedding model id (must match index)")
    parser.add_argument("--embed-batch-size", type=int, default=0, help="Batch size for encoding queries (0=process all in one batch)")
    parser.add_argument("--embed-tensor-parallel-size", type=int, default=8, help="Tensor parallel size for embedding engine (vLLM)")
    parser.add_argument("--embed-dtype", default="auto", help="dtype for embedding engine (e.g., auto, float16, bfloat16)")

    # Generation controls
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--show-reasoning", action="store_true", help="Print reasoning content if provided by endpoint")
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B", help="HF model id to load with vLLM backend")
    parser.add_argument("--tensor-parallel-size", type=int, default=8, help="Tensor parallel size for generation engine (vLLM)")
    parser.add_argument("--gen-dtype", default="auto", help="dtype for generation engine (e.g., auto, float16, bfloat16)")
    parser.add_argument("--embed-gpu-mem-frac", type=float, default=0.1, help="GPU memory fraction for embedding engine (e.g., 0.1)")
    parser.add_argument("--gen-gpu-mem-frac", type=float, default=0.8, help="GPU memory fraction for generation engine (e.g., 0.8)")

    # Outputs
    parser.add_argument("--output-json", default="", help="Optional path to write batch results as JSON")

    args = parser.parse_args()

    # Load queries
    if args.query_file:
        queries = read_batch_prompts(args.query_file)
    elif args.query:
        queries = [args.query]

    # Build row metadata in the exact index order
    prompts_map = build_prompts_map_from_dataset(args.data_file)
    texts_in_order, prompt_infos_in_order, row_records = build_texts_and_prompt_shas(
        args.experiences_dir, prompts_map
    )

    # Load index
    emb_path = os.path.join(args.embed_index_dir, "embeddings.npy")
    prompt_info_path = os.path.join(args.embed_index_dir, "prompt_sha1.json")
    meta_path = os.path.join(args.embed_index_dir, "metadata.json")
    assert os.path.exists(emb_path) and os.path.exists(prompt_info_path) and os.path.exists(meta_path), "Missing index files"
    doc_emb = np.load(emb_path)
    with open(prompt_info_path, "r", encoding="utf-8") as f:
        prompt_infos_saved: list[dict] = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Sanity checks
    assert doc_emb.ndim == 2, "embeddings.npy must be 2D"
    assert len(prompt_infos_saved) == doc_emb.shape[0] == len(row_records), "Index misalignment"

    # Build embedding engine and encode queries
    embed_engine = LLM(
        model=args.embed_model_id,
        task="embed",
        trust_remote_code=True,
        tensor_parallel_size=int(args.embed_tensor_parallel_size),
        dtype=args.embed_dtype,
        gpu_memory_utilization=args.embed_gpu_mem_frac,
    )
    embed_bs = args.embed_batch_size if args.embed_batch_size and args.embed_batch_size > 0 else len(queries)
    query_vecs = encode_with_vllm(embed_engine, queries, batch_size=max(1, embed_bs))

    # Retrieval per query
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompt_texts: list[str] = []
    meta_out: list[dict] = []

    # Ensure doc_emb is normalized; if not, normalize defensively
    norms = np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-12
    doc_emb_norm = doc_emb / norms

    for qi, qv in enumerate(query_vecs):
        scores = doc_emb_norm @ qv.reshape(-1, 1)
        scores = scores.reshape(-1)
        k = int(args.top_k)
        n = len(scores)
        k = min(k, n)
        order_desc = np.argsort(-scores)
        if k == n:
            idx = order_desc
        else:
            kth_score = float(scores[order_desc[k - 1]])
            # Strictly higher than the kth score keep their relative (desc) order
            above_mask = scores > kth_score
            above_indices = np.where(above_mask)[0]
            above_indices = order_desc[np.isin(order_desc, above_indices)]

            need = k - len(above_indices)
            if need > 0:
                # Randomly sample from the tied group at the cutoff
                tie_mask = np.isclose(scores, kth_score, rtol=1e-6, atol=1e-8)
                tie_indices = np.where(tie_mask)[0]
                if len(tie_indices) > 0:
                    sampled = np.random.choice(tie_indices, size=min(need, len(tie_indices)), replace=False)
                    idx = np.concatenate([above_indices, sampled])
                else:
                    idx = above_indices[:k]
            else:
                idx = above_indices[:k]

        # Build retrieved rows with prompt + experience
        retrieved_rows: list[dict] = []
        for ridx in idx:
            rr = row_records[ridx]
            pinfo = prompt_infos_saved[ridx]
            prompt_txt = (pinfo.get("prompt") or "").strip()
            full_txt = rr.text or ""
            # Embedding rows store "prompt + experiences"; strip the prompt prefix for display
            exp_only = full_txt
            if prompt_txt:
                if full_txt.startswith(prompt_txt + " "):
                    exp_only = full_txt[len(prompt_txt) + 1 :]
                elif full_txt.startswith(prompt_txt):
                    exp_only = full_txt[len(prompt_txt) :].lstrip()
            retrieved_rows.append(
                {
                    "score": float(scores[ridx]),
                    "problem_key": rr.problem_key,
                    "experience_id": rr.experience_id,
                    "prompt": prompt_txt,
                    "experience": exp_only,
                }
            )

        experiences_block = _format_experiences_for_prompt(retrieved_rows)
        user_prompt = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
            problem=queries[qi],
            experiences=experiences_block,
        )
        prompt_texts.append(_to_qwen_thinking_chat(user_prompt, tokenizer))
        meta_out.append({"prompt": queries[qi]})

    # Generate in batch with vLLM
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    vllm_llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        tensor_parallel_size=int(args.tensor_parallel_size),
        dtype=args.gen_dtype,
        gpu_memory_utilization=args.gen_gpu_mem_frac,
    )
    
    # Optionally preview the first final prompt for a batch
    try:
        print("FIRST_FINAL_PROMPT:\n" + str(prompt_texts[0]), file=sys.stderr)
    except Exception:
        pass
    outputs = vllm_llm.generate(prompt_texts, sampling_params)

    # Collect results
    results = []
    for i, out in enumerate(outputs):
        raw = out.outputs[0].text if out.outputs else ""
        content = _extract_final_answer(raw) or raw
        item = {
            "prompt": meta_out[i]["prompt"],
            "content": content,
        }
        if args.show_reasoning:
            item["raw"] = raw
        results.append(item)

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

