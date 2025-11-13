import json
from pathlib import Path

NB_PATH = str(Path(__file__).resolve().parents[1] / "grpo_vllm_semiconductor.ipynb")

MARK = "# === Evaluate NEW, compute deltas vs OLD, update scores, filter NEW, sync OLD ===\n"
TRY_START = "            try:\n"
TRY_END = "            except Exception as __e:\n                print(f\"[Step {step}] WARNING: NEW evaluation/filtering failed: {__e}\")\n"
 
NEW_BLOCK = (
    "            # === Evaluate NEW vs OLD per problem; update OLD only when delta >= 0 ===\n"
    "            try:\n"
    "                # 1) Evaluate NEW using full retrieved experience sets (no globals)\n"
    "                eval_problem_totals = {}\n"
    "                GRPO_N_EVAL = 1\n"
    "                EVAL_TEMPERATURE = 0.1\n"
    "\n"
    "                for _batch_idx in range(num_batches):\n"
    "                    _start = _batch_idx * BATCH_SIZE\n"
    "                    _end = min(len(data), (_batch_idx + 1) * BATCH_SIZE)\n"
    "                    _idx_slice = step_indices[_start:_end]\n"
    "                    _batch_data = [data[i] for i in _idx_slice]\n"
    "\n"
    "                    _problems = [s['problem'] for s in _batch_data]\n"
    "                    _keys = [compute_problem_key(p) for p in _problems]\n"
    "\n"
    "                    _retrieve_k = max(TOP_K_EXPERIENCES * 3, TOP_K_EXPERIENCES)\n"
    "                    _retrieved_lists = run_search(\n"
    "                        experiences_dir=EXPERIENCES_NEW_ROOT,\n"
    "                        query=_problems,\n"
    "                        top_k=_retrieve_k,\n"
    "                        batch_size=EMBED_BATCH_SIZE,\n"
    "                        instruction=EMBED_INSTRUCTION,\n"
    "                        index_dir=EXPERIENCES_INDEX_DIR,\n"
    "                        engine=embed_engine,\n"
    "                    )\n"
    "\n"
    "                    _formatted_batch = []\n"
    "                    for _sample, _problem_text, _current_key, _retrieved in zip(_batch_data, _problems, _keys, _retrieved_lists or []):\n"
    "                        _filtered = [it for it in (_retrieved or []) if it.get('problem_key') != _current_key]\n"
    "                        _top_items = _filtered[:TOP_K_EXPERIENCES]\n"
    "                        _prompt = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(\n"
    "                            experiences=format_experiences_for_prompt(_top_items),\n"
    "                            problem=_problem_text,\n"
    "                        )\n"
    "                        _formatted_batch.append({\"prompt\": _prompt, **_sample})\n"
    "\n"
    "                    _formatted_batch = _formatted_batch * GRPO_N_EVAL\n"
    "                    _prompts = [x[\"prompt\"] for x in _formatted_batch]\n"
    "                    _chat_prompts = [to_qwen_thinking_chat(p, gen_tokenizer) for p in _prompts]\n"
    "                    _eval_sampling_params = SamplingParams(\n"
    "                        temperature=EVAL_TEMPERATURE,\n"
    "                        max_tokens=MAX_NEW_TOKENS,\n"
    "                        top_p=0.95,\n"
    "                        top_k=20,\n"
    "                    )\n"
    "                    _eval_outs = gen.generate(_chat_prompts, _eval_sampling_params)\n"
    "                    _eval_texts = [extract_final_answer(o.outputs[0].text) for o in _eval_outs]\n"
    "\n"
    "                    _problem_to_rollouts_eval = {}\n"
    "                    for _item, _out in zip(_formatted_batch, _eval_texts):\n"
    "                        _r = dict(_item); _r[\"response\"] = _out; _r[\"reward\"] = 0.0\n"
    "                        _problem_to_rollouts_eval.setdefault(_r[\"problem\"], []).append(_r)\n"
    "\n"
    "                    _grading_prompts_eval, _grading_refs_eval = [], []\n"
    "                    for _problem, _rs in _problem_to_rollouts_eval.items():\n"
    "                        _req_block = _rs[0].get(\"keypoints\")\n"
    "                        _req_text = format_requirements_block(_req_block) if isinstance(_req_block, list) else \"\"\n"
    "                        for _i, _each in enumerate(_rs):\n"
    "                            _grading_prompts_eval.append(\n"
    "                                SINGLE_ROLLOUT_GRADING_TEMPLATE.format(\n"
    "                                    problem=_problem, response=str(_each.get(\"response\", \"\")), requirements=_req_text\n"
    "                                )\n"
    "                            )\n"
    "                            _grading_refs_eval.append((_problem, _i, _each))\n"
    "\n"
    "                    _grading_chat_prompts_eval = [to_qwen_thinking_chat(p, grade_tokenizer) for p in _grading_prompts_eval]\n"
    "                    _g_params_eval = SamplingParams(\n"
    "                        temperature=GRADING_TEMPERATURE,\n"
    "                        max_tokens=GRADING_MAX_NEW_TOKENS,\n"
    "                        top_p=0.95,\n"
    "                        top_k=20,\n"
    "                    )\n"
    "                    _g_outs_eval = grade_gen.generate(_grading_chat_prompts_eval, _g_params_eval)\n"
    "\n"
    "                    for (_problem, _i, _each), _gout in zip(_grading_refs_eval, _g_outs_eval):\n"
    "                        _gtxt = extract_final_answer(_gout.outputs[0].text) if _gout and _gout.outputs else \"\"\n"
    "                        _gjson = safe_json_obj(_gtxt)\n"
    "                        _total_grade = 0.0\n"
    "                        try:\n"
    "                            for _v in (_gjson or {}).values():\n"
    "                                _g = _v.get(\"grade\") if isinstance(_v, dict) else None\n"
    "                                if isinstance(_g, (int, float)):\n"
    "                                    _total_grade += float(_g)\n"
    "                                elif isinstance(_g, str):\n"
    "                                    _digits = \"\".join(ch for ch in _g if ch.isdigit())\n"
    "                                    if _digits:\n"
    "                                        _total_grade += float(int(_digits))\n"
    "                        except Exception:\n"
    "                            pass\n"
    "                        _kp = _each.get(\"keypoints\") or []\n"
    "                        _num_points = max(len(_kp) - 1, 1) if isinstance(_kp, list) else 1\n"
    "                        _denom = 4.0 * float(_num_points)\n"
    "                        _each[\"reward\"] = (_total_grade / _denom) if _denom > 0 else 0.0\n"
    "\n"
    "                    for _p, _rs in _problem_to_rollouts_eval.items():\n"
    "                        _pkey = compute_problem_key(_p)\n"
    "                        eval_problem_totals[_pkey] = eval_problem_totals.get(_pkey, 0.0) + sum(x[\"reward\"] for x in _rs)\n"
    "\n"
    "                # 2) Compute per-problem delta (NEW - OLD)\n"
    "                per_problem_delta = {}\n"
    "                for _pkey in set(list(eval_problem_totals.keys()) + list(cur_problem_totals.keys())):\n"
    "                    _new_total = float(eval_problem_totals.get(_pkey, 0.0))\n"
    "                    _old_total = float(cur_problem_totals.get(_pkey, 0.0))\n"
    "                    per_problem_delta[_pkey] = _new_total - _old_total\n"
    "\n"
    "                # 3) Selectively update OLD with NEW only when delta >= 0 for that problem\n"
    "                updated, skipped = 0, 0\n"
    "                for _src in sorted(glob.glob(os.path.join(EXPERIENCES_NEW_ROOT, '*.json'))):\n"
    "                    _pkey = os.path.splitext(os.path.basename(_src))[0]\n"
    "                    _delta = float(per_problem_delta.get(_pkey, 0.0))\n"
    "                    if _delta >= 0.0:\n"
    "                        _dst = os.path.join(EXPERIENCES_OLD_ROOT, os.path.basename(_src))\n"
    "                        os.makedirs(os.path.dirname(_dst), exist_ok=True)\n"
    "                        from shutil import copy2\n"
    "                        copy2(_src, _dst)\n"
    "                        updated += 1\n"
    "                    else:\n"
    "                        skipped += 1\n"
    "                print(f\"[Step {step}] OLD selectively updated from NEW: updated={updated}, skipped={skipped}\")\n"
    "\n"
    "            except Exception as __e:\n"
    "                print(f\"[Step {step}] WARNING: NEW evaluation/selective update failed: {__e}\")\n"
)


def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        if not isinstance(src, list):
            continue
        text = "".join(src)
        if MARK not in text:
            continue
        start_idx = text.find(MARK)
        try_idx = text.find(TRY_START, start_idx)
        end_idx = text.find(TRY_END, try_idx)
        if start_idx == -1 or try_idx == -1 or end_idx == -1:
            continue
        end_idx += len(TRY_END)
        new_text = text[:start_idx] + NEW_BLOCK + text[end_idx:]
        c["source"] = new_text.splitlines(True)
        with open(NB_PATH, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print("Notebook updated for per-problem selective update of OLD.")
        return

    print("Marker not found; no changes made.")

if __name__ == "__main__":
    main()
