import json
import re
from pathlib import Path

NB_PATH = str(Path(__file__).resolve().parents[1] / "grpo_vllm_semiconductor.ipynb")


def remove_between(text: str, start_pat: str, end_pat: str, include_end=False) -> str:
    s = text.find(start_pat)
    if s == -1:
        return text
    e = text.find(end_pat, s)
    if e == -1:
        return text
    e2 = e + (len(end_pat) if include_end else 0)
    return text[:s] + text[e2:]


def main():
    nb = json.loads(Path(NB_PATH).read_text(encoding="utf-8"))

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src)
        if "# ---- Multi-step rollout & experience update ----" not in text:
            continue

        # 1) Remove prev_step_total usage
        text = text.replace("prev_step_total = None\n", "")
        text = text.replace("            prev_step_total = step_total\n", "")

        # 2) Remove grade_totals persistence block (anchor by the comment)
        text = remove_between(
            text,
            "            # 5) Save step total and advance baseline metric\n",
            "\n            prev_step_total = step_total\n",
        )
        # In case the trailing assignment was already removed, also try removing until the print
        text = remove_between(
            text,
            "            # 5) Save step total and advance baseline metric\n",
            "\n        print(f\"[Step {step}][Batch {batch_idx+1}/{num_batches}] Experiences updated.\")\n",
        )

        # 3) Simplify scoring state: drop prev_problem_totals/exp_scores/retrieved_map
        start_marker = "    # Scoring: load previous per-problem totals and cumulative experience scores\n"
        sidx = text.find(start_marker)
        if sidx != -1:
            # Find the line where cur_problem_totals is declared
            cur_decl = "    cur_problem_totals: Dict[str, float] = {}\n"
            cidx = text.find(cur_decl, sidx)
            if cidx != -1:
                # Replace entire block from start_marker to end of retrieved_map decl with just cur_problem_totals
                # Remove retrieved_map line if present
                after_block = text[cidx + len(cur_decl):]
                after_block = after_block.replace("    retrieved_map: Dict[str, Set[str]] = {}\n", "")
                # Now rebuild text
                text = text[:sidx] + "    # Scoring state\n" + cur_decl + after_block

        # 4) Remove per-sample used_uids tracking block
        used_uids_start = (
            "            # Track retrieved experience UIDs for scoring\n"
        )
        used_uids_end_anchor = "            formatted_experiences = format_experiences_for_prompt(top_items)\n"
        text = remove_between(text, used_uids_start, used_uids_end_anchor)

        cell["source"] = text.splitlines(True)
        break

    Path(NB_PATH).write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Notebook further simplified: removed unused scoring state and totals persistence.")


if __name__ == "__main__":
    main()
