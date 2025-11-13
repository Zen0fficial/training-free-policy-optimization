import json
import os
import re
from pathlib import Path

NB_PATH = str(Path(__file__).resolve().parents[1] / "grpo_vllm_semiconductor.ipynb")

# Markers and snippets
MAIN_BLOCK_MARK = "# ---- Multi-step rollout & experience update ----"
RETR_CALL_PATTERN = "experiences_dir=EXPERIENCES_NEW_ROOT"
RETR_CALL_REPLACEMENT = "experiences_dir=EXPERIENCES_OLD_ROOT"
REBUILD_ARG_PATTERN = ",\n            rebuild=True,\n"
REBUILD_ARG_REPLACEMENT = "\n"

SCORES_BLOCK_START = "            # 2) Compute experience scores from per-problem deltas"
ADOPTION_BLOCK_START = "            # 3) Compare totals and decide adoption"
ADOPTION_BLOCK_END_GUARD = "            # 4) Rebuild NEW from baseline (updated OLD if adopted, else existing OLD), then apply ops"

# Block to insert after NEW rebuilt message
NEW_EVAL_BLOCK = """
            # === Evaluate NEW, compute deltas vs OLD, update scores, filter NEW, sync OLD ===
            try:
                # 1) NEW-pass evaluation
                from collections import defaultdict
                os.environ["EXPERIENCES_FORCE_DIR"] = EXPERIENCES_NEW_ROOT
                eval_problem_totals = {}
                retrieved_map_new = defaultdict(set)

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
                        batch_size=EMBED_BATCH_SIZE,
                        instruction=EMBED_INSTRUCTION,
                        index_dir=EXPERIENCES_INDEX_DIR,
                        engine=embed_engine,
                    )

                    _formatted_batch = []
                    for _sample, _problem_text, _current_key, _retrieved in zip(_batch_data, _problems, _keys, _retrieved_lists or []):
                        _filtered = [it for it in (_retrieved or []) if it.get('problem_key') != _current_key]
                        _top_items = _filtered[:TOP_K_EXPERIENCES]

                        _used_uids = set()
                        for it in _top_items:
                            _txt = (it.get('text') or '')
                            _t_sha = hashlib.sha1(_txt.encode('utf-8')).hexdigest()[:12]
                            _uid = f"{it.get('problem_key')}::{it.get('experience_id')}::{_t_sha}"
                            _used_uids.add(_uid)
                        if _used_uids:
                            retrieved_map_new[_current_key].update(_used_uids)

                        _prompt = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                            experiences=format_experiences_for_prompt(_top_items),
                            problem=_problem_text,
                        )
                        _formatted_batch.append({"prompt": _prompt, **_sample})

                    _formatted_batch = _formatted_batch * GRPO_N_EVAL
                    _prompts = [x["prompt"] for x in _formatted_batch]
                    _chat_prompts = [to_qwen_thinking_chat(p, gen_tokenizer) for p in _prompts]
                    _eval_sampling_params = SamplingParams(
                        temperature=EVAL_TEMPERATURE,
                        max_tokens=MAX_NEW_TOKENS,
                        top_p=0.95,
                        top_k=20,
                    )
                    _eval_outs = gen.generate(_chat_prompts, _eval_sampling_params)
                    _eval_texts = [extract_final_answer(o.outputs[0].text) for o in _eval_outs]

                    _problem_to_rollouts_eval = {}
                    for _item, _out in zip(_formatted_batch, _eval_texts):
                        _r = dict(_item); _r["response"] = _out; _r["reward"] = 0.0
                        _problem_to_rollouts_eval.setdefault(_r["problem"], []).append(_r)

                    _grading_prompts_eval, _grading_refs_eval = [], []
                    for _problem, _rs in _problem_to_rollouts_eval.items():
                        _req_block = _rs[0].get("keypoints")
                        _req_text = format_requirements_block(_req_block) if isinstance(_req_block, list) else ""
                        for _i, _each in enumerate(_rs):
                            _grading_prompts_eval.append(
                                SINGLE_ROLLOUT_GRADING_TEMPLATE.format(
                                    problem=_problem, response=str(_each.get("response", "")), requirements=_req_text
                                )
                            )
                            _grading_refs_eval.append((_problem, _i, _each))

                    _grading_chat_prompts_eval = [to_qwen_thinking_chat(p, grade_tokenizer) for p in _grading_prompts_eval]
                    _g_params_eval = SamplingParams(
                        temperature=GRADING_TEMPERATURE,
                        max_tokens=GRADING_MAX_NEW_TOKENS,
                        top_p=0.95,
                        top_k=20,
                    )
                    _g_outs_eval = grade_gen.generate(_grading_chat_prompts_eval, _g_params_eval)

                    for (_problem, _i, _each), _gout in zip(_grading_refs_eval, _g_outs_eval):
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
                        _kp = _each.get("keypoints") or []
                        _num_points = max(len(_kp) - 1, 1) if isinstance(_kp, list) else 1
                        _denom = 4.0 * float(_num_points)
                        _each["reward"] = (_total_grade / _denom) if _denom > 0 else 0.0

                    for _p, _rs in _problem_to_rollouts_eval.items():
                        _pkey = compute_problem_key(_p)
                        eval_problem_totals[_pkey] = eval_problem_totals.get(_pkey, 0.0) + sum(x["reward"] for x in _rs)

                # 2) Delta per problem: NEW - OLD (cur_problem_totals is OLD pass totals from main loop)
                per_problem_delta = {}
                for _pkey in set(list(eval_problem_totals.keys()) + list(cur_problem_totals.keys())):
                    _new_total = float(eval_problem_totals.get(_pkey, 0.0))
                    _old_total = float(cur_problem_totals.get(_pkey, 0.0))
                    per_problem_delta[_pkey] = _new_total - _old_total

                # 3) Update experience scores for NEW UIDs
                exp_scores_path = os.path.join(SCORES_DIR, "exp_scores.json")
                try:
                    exp_scores = json.load(open(exp_scores_path, "r", encoding="utf-8"))
                    if not isinstance(exp_scores, dict):
                        exp_scores = {}
                except Exception:
                    exp_scores = {}

                for _pkey, _used_uids in retrieved_map_new.items():
                    _delta = float(per_problem_delta.get(_pkey, 0.0))
                    for _uid in _used_uids:
                        _row = exp_scores.get(_uid) or {"problem_key": "", "experience_id": "", "text_sha1": "", "sum_delta": 0.0, "count": 0}
                        _parts = _uid.split("::")
                        if not _row.get("problem_key"): _row["problem_key"] = _parts[0] if len(_parts) > 0 else ""
                        if not _row.get("experience_id"): _row["experience_id"] = _parts[1] if len(_parts) > 1 else ""
                        if not _row.get("text_sha1"): _row["text_sha1"] = _parts[2] if len(_parts) > 2 else ""
                        _row["sum_delta"] = float(_row.get("sum_delta", 0.0)) + _delta
                        _row["count"] = int(_row.get("count", 0)) + 1
                        _row["avg_delta"] = _row["sum_delta"] / max(1, _row["count"])
                        exp_scores[_uid] = _row

                with open(exp_scores_path, "w", encoding="utf-8") as f:
                    json.dump(exp_scores, f, ensure_ascii=False, indent=2)

                # 4) Filter NEW in-place: keep avg_delta >= 0.0
                def _uid(_problem_key, _exp_id, _text):
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
                            _u = _uid(_pkey, _k, _v)
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

                # 5) Sync OLD <- filtered NEW
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
                except Exception as __e:
                    print(f"[Step {step}] WARNING: failed syncing filtered NEW->OLD: {__e}")

                # Restore OLD for next main rollout
                os.environ["EXPERIENCES_FORCE_DIR"] = EXPERIENCES_OLD_ROOT
            except Exception as __e:
                print(f"[Step {step}] WARNING: NEW evaluation/filtering failed: {__e}")
"""

def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    changed = False

    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        if not isinstance(src, list):
            continue
        text = "".join(src)
        if MAIN_BLOCK_MARK not in text:
            continue
        # Retrieval: NEW -> OLD
        new_text = text.replace(RETR_CALL_PATTERN, RETR_CALL_REPLACEMENT)
        # Remove rebuild=True in index build
        new_text = new_text.replace(REBUILD_ARG_PATTERN, REBUILD_ARG_REPLACEMENT)

        # Remove cross-step scoring and adoption blocks
        s_idx = new_text.find(SCORES_BLOCK_START)
        a_idx = new_text.find(ADOPTION_BLOCK_START)
        guard_idx = new_text.find(ADOPTION_BLOCK_END_GUARD)
        if s_idx != -1 and guard_idx != -1:
            # Cut from SCORES block start up to just before the guard marker (keep guard marker line)
            before = new_text[:s_idx]
            after = new_text[guard_idx:]
            # Now in the 'after' block, remove the adoption block content up to the guard marker line
            # Adoption block is between a_idx and guard_idx; already removed by taking 'after' from guard_idx
            new_text = before + after
            changed = True

        # Insert NEW evaluation block after the "NEW experiences rebuilt ..." line
        insert_anchor = "                print(f\"[Step {step}] NEW experiences rebuilt from baseline with ops applied.\")\n"
        if insert_anchor in new_text and NEW_EVAL_BLOCK not in new_text:
            new_text = new_text.replace(insert_anchor, insert_anchor + NEW_EVAL_BLOCK)
            changed = True

        if changed:
            c["source"] = new_text.splitlines(True)
            break

    if changed:
        with open(NB_PATH, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print("Notebook patched successfully.")
    else:
        print("No changes applied (markers not found or already patched).")


if __name__ == "__main__":
    main()
