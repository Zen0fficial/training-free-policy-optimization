import json
from pathlib import Path

NB_PATH = str(Path(__file__).resolve().parents[1] / "grpo_vllm_semiconductor.ipynb")

MARK = "# === Evaluate NEW, compute deltas vs OLD, update scores, filter NEW, sync OLD ===\n"
TRY_START = "            try:\n"
TRY_END = "            except Exception as __e:\n                print(f\"[Step {step}] WARNING: NEW evaluation/filtering failed: {__e}\")\n"

NEW_BLOCK = (
    MARK
    + TRY_START
    + """
                # 1) NEW-pass individual experience evaluation (no globals)
                eval_problem_baseline = dict(cur_problem_totals)  # OLD baseline from main pass
                exp_stats = {}  # uid -> {sum_delta, count}

                GRPO_N_EVAL = 1
                EVAL_TEMPERATURE = 0.1

                for _batch_idx in range(num_batches):
                    _start = _batch_idx * BATCH_SIZE
                    _end = min(len(data), (_batch_idx + 1) * BATCH_SIZE)
                    _idx_slice = step_indices[_start:_end]
                    _batch_data = [data[i] for i in _idx_slice]

                    _problems = [s['problem'] for s in _batch_data]
                    _keys = [compute_problem_key(p) for p in _problems]

                    _retrieve_k = max(TOP_K_EXPERIENCES * 3, TOP_K_EXPERIENCES)
                    _retrieved_lists = run_search(
                        experiences_dir=EXPERIENCES_NEW_ROOT,
                        query=_problems,
                        top_k=_retrieve_k,
                        device=None,
                        batch_size=EMBED_BATCH_SIZE,
                        use_flash_attention_2=False,
                        instruction=EMBED_INSTRUCTION,
                        index_dir=EXPERIENCES_INDEX_DIR,
                        rebuild_index=False,
                        engine=embed_engine,
                    )

                    # Build one prompt per (problem, single experience)
                    _single_prompts = []
                    _single_refs = []  # (pkey, uid)
                    for _problem_text, _current_key, _retrieved in zip(_problems, _keys, _retrieved_lists or []):
                        _filtered = [it for it in (_retrieved or []) if it.get('problem_key') != _current_key]
                        _top_items = _filtered[:TOP_K_EXPERIENCES]
                        for it in _top_items:
                            _txt = (it.get('text') or '')
                            _t_sha = hashlib.sha1(_txt.encode('utf-8')).hexdigest()[:12]
                            _uid = f"{it.get('problem_key')}::{it.get('experience_id')}::{_t_sha}"
                            _prompt = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                                experiences=format_experiences_for_prompt([it]),
                                problem=_problem_text,
                            )
                            _single_prompts.append(_prompt)
                            _single_refs.append((_current_key, _uid))

                    if not _single_prompts:
                        continue

                    _chat_prompts = [to_qwen_thinking_chat(p, gen_tokenizer) for p in _single_prompts]
                    _eval_sampling_params = SamplingParams(
                        temperature=EVAL_TEMPERATURE,
                        max_tokens=MAX_NEW_TOKENS,
                        top_p=0.95,
                        top_k=20,
                    )
                    _eval_outs = gen.generate(_chat_prompts, _eval_sampling_params)
                    _eval_texts = [extract_final_answer(o.outputs[0].text) for o in _eval_outs]

                    # Grade each single-exp prompt
                    _grading_prompts_eval, _grading_refs_eval = [], []
                    for (_pkey, _uid), _resp in zip(_single_refs, _eval_texts):
                        _req_block = (_batch_data[0].get("keypoints") if _batch_data else [])
                        _req_text = format_requirements_block(_req_block) if isinstance(_req_block, list) else ""
                        _grading_prompts_eval.append(
                            SINGLE_ROLLOUT_GRADING_TEMPLATE.format(
                                problem="", response=str(_resp), requirements=_req_text
                            )
                        )
                        _grading_refs_eval.append((_pkey, _uid))

                    _grading_chat_prompts_eval = [to_qwen_thinking_chat(p, grade_tokenizer) for p in _grading_prompts_eval]
                    _g_params_eval = SamplingParams(
                        temperature=GRADING_TEMPERATURE,
                        max_tokens=GRADING_MAX_NEW_TOKENS,
                        top_p=0.95,
                        top_k=20,
                    )
                    _g_outs_eval = grade_gen.generate(_grading_chat_prompts_eval, _g_params_eval)

                    # Assign per-experience deltas
                    for (_pkey, _uid), _gout in zip(_grading_refs_eval, _g_outs_eval):
                        _gtxt = extract_final_answer(_gout.outputs[0].text) if _gout and _gout.outputs else ""
                        _gjson = safe_json_obj(_gtxt)
                        _total_grade = 0.0
                        try:
                            for _v in (_gjson or {}).values():
                                _g = _v.get("grade") if isinstance(_v, dict) else None
                                if isinstance(_g, (int, float)):
                                    _total_grade += float(_g)
                                elif isinstance(_g, str):
                                    _digits = "".join(ch for ch in _g if ch.isdigit())
                                    if _digits:
                                        _total_grade += float(int(_digits))
                        except Exception:
                            pass
                        # Normalize by number of keypoints (use the same scheme)
                        _kp = _batch_data[0].get("keypoints") if _batch_data else []
                        _num_points = max(len(_kp) - 1, 1) if isinstance(_kp, list) else 1
                        _denom = 4.0 * float(_num_points)
                        _reward = (_total_grade / _denom) if _denom > 0 else 0.0
                        _baseline = float(eval_problem_baseline.get(_pkey, 0.0))
                        _delta = _reward - _baseline
                        _row = exp_stats.get(_uid) or {"sum_delta": 0.0, "count": 0}
                        _row["sum_delta"] += _delta
                        _row["count"] += 1
                        exp_stats[_uid] = _row

                # 2) Persist per-experience averages
                exp_scores_path = os.path.join(SCORES_DIR, "exp_scores.json")
                try:
                    exp_scores = json.load(open(exp_scores_path, "r", encoding="utf-8"))
                    if not isinstance(exp_scores, dict):
                        exp_scores = {}
                except Exception:
                    exp_scores = {}

                for _uid, _row in exp_stats.items():
                    _cur = exp_scores.get(_uid) or {"sum_delta": 0.0, "count": 0}
                    _cur["sum_delta"] = float(_cur.get("sum_delta", 0.0)) + float(_row["sum_delta"])
                    _cur["count"] = int(_cur.get("count", 0)) + int(_row["count"])
                    _cur["avg_delta"] = _cur["sum_delta"] / max(1, _cur["count"])
                    exp_scores[_uid] = _cur

                with open(exp_scores_path, "w", encoding="utf-8") as f:
                    json.dump(exp_scores, f, ensure_ascii=False, indent=2)

                # 3) Filter NEW in-place: keep avg_delta >= 0.0
                def _uid_compute(_problem_key, _exp_id, _text):
                    _t_sha = hashlib.sha1((_text or "").encode("utf-8")).hexdigest()[:12]
                    return f"{_problem_key}::{str(_exp_id)}::{_t_sha}"

                _files = sorted(glob.glob(os.path.join(EXPERIENCES_NEW_ROOT, "*.json")))
                _files_updated, _removed, _kept = 0, 0, 0
                for _fp in _files:
                    try:
                        _pkey = os.path.splitext(os.path.basename(_fp))[0]
                        _obj = json.load(open(_fp, "r", encoding="utf-8"))
                        if not isinstance(_obj, dict) or not _obj:
                            continue
                        _out = {}
                        for _k, _v in _obj.items():
                            _u = _uid_compute(_pkey, _k, _v)
                            _row = exp_scores.get(_u) if isinstance(exp_scores, dict) else None
                            _score = 0.0
                            if isinstance(_row, dict):
                                try:
                                    _score = float(_row.get("avg_delta", 0.0))
                                except Exception:
                                    _score = 0.0
                            if _score >= 0.0:
                                _out[str(_k)] = str(_v); _kept += 1
                            else:
                                _removed += 1
                        if _out != _obj:
                            json.dump(_out, open(_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                            _files_updated += 1
                    except Exception:
                        continue
                print(f"[Step {step}] NEW filtered: files_updated={_files_updated}, kept={_kept}, removed={_removed}")

                # 4) Sync OLD <- filtered NEW
                try:
                    try:
                        shutil.rmtree(EXPERIENCES_OLD_ROOT)
                    except Exception:
                        pass
                    os.makedirs(EXPERIENCES_OLD_ROOT, exist_ok=True)
                    for _src in glob.glob(os.path.join(EXPERIENCES_NEW_ROOT, "*.json")):
                        _dst = os.path.join(EXPERIENCES_OLD_ROOT, os.path.basename(_src))
                        shutil.copy2(_src, _dst)
                    print(f"[Step {step}] OLD replaced with filtered NEW.")
                except Exception as __sync_e:
                    print(f"[Step {step}] WARNING: failed syncing filtered NEW->OLD: {__sync_e}")
    """
    + TRY_END
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
        # Find current try/except block starting at marker
        start_idx = text.find(MARK)
        try_idx = text.find(TRY_START, start_idx)
        end_idx = text.find(TRY_END, try_idx) + len(TRY_END)
        if start_idx == -1 or try_idx == -1 or end_idx == -1:
            continue
        new_text = text[:start_idx] + NEW_BLOCK + text[end_idx:]
        c["source"] = new_text.splitlines(True)
        with open(NB_PATH, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print("Notebook updated with per-experience scoring and no globals.")
        return

    print("Marker not found; no changes made.")

if __name__ == "__main__":
    main()
