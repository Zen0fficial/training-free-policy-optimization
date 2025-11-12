import json
import re
from pathlib import Path

NB_PATH = str(Path(__file__).resolve().parents[1] / "grpo_vllm_semiconductor.ipynb")

PRE_START = "    # Precompute embeddings for NEW experiences at the beginning of this step\n"
TRY_LINE = "    try:\n"
EXCEPT_LINE = "    except Exception as e:\n"
EXCEPT_PRINT_PREFIX = "        print(f\"[Step {step}] WARNING: failed to precompute embeddings: "

OPS_ANCHOR = "    # Recreate ops dir for this step\n"
OPS_TRY_BLOCK = (
    "    try:\n"
    "        try:\n"
    "            shutil.rmtree(ops_dir)\n"
    "        except Exception:\n"
    "            pass\n"
    "        os.makedirs(ops_dir, exist_ok=True)\n"
    "    except Exception as _e:\n"
    "        print(f\"[Step {step}] WARNING: failed to init ops dir: {_e}\")\n"
)
OPS_REPLACEMENT = (
    "    shutil.rmtree(ops_dir, ignore_errors=True)\n"
    "    os.makedirs(ops_dir, exist_ok=True)\n"
)

def dedent_block(block_text: str, spaces: int = 4) -> str:
    out_lines = []
    for ln in block_text.splitlines(True):
        if ln.startswith(" " * spaces):
            out_lines.append(ln[spaces:])
        else:
            out_lines.append(ln)
    return "".join(out_lines)


def main():
    nb = json.loads(Path(NB_PATH).read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src)

        # 1) Remove try/except around precompute embeddings
        pre_idx = text.find(PRE_START)
        if pre_idx != -1:
            try_idx = text.find(TRY_LINE, pre_idx)
            if try_idx != -1:
                except_idx = text.find(EXCEPT_LINE, try_idx)
                if except_idx != -1:
                    # Content between try and except
                    content = text[try_idx + len(TRY_LINE):except_idx]
                    content_dedented = dedent_block(content, 4)
                    # Find end of except print line
                    # We will drop the entire except block (two lines)
                    after_except_idx = text.find("\n", except_idx)
                    after_print_idx = text.find("\n", after_except_idx + 1)
                    if after_print_idx == -1:
                        after_print_idx = len(text)
                    new_text = text[:try_idx] + content_dedented + text[after_print_idx + 1:]
                    text = new_text

        # 2) Simplify ops dir init block
        if OPS_TRY_BLOCK in text:
            text = text.replace(OPS_TRY_BLOCK, OPS_REPLACEMENT)

        cell["source"] = text.splitlines(True)

    Path(NB_PATH).write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Notebook simplified: removed unnecessary try/except blocks.")


if __name__ == "__main__":
    main()
