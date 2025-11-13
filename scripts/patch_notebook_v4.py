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


def ensure_llm_import(text: str) -> str:
    """Ensure the cell has `from vllm import LLM`. If a vllm import exists but lacks LLM, append it."""
    lines = text.splitlines(True)
    for i, ln in enumerate(lines):
        if ln.startswith("from vllm import"):
            if "LLM" not in ln:
                # add LLM to the existing import line
                if ln.rstrip().endswith("\\"):
                    # handle line continuation conservatively by appending before backslash
                    lines[i] = ln.rstrip("\n").rstrip()
                    if lines[i].endswith("\\"):
                        lines[i] = lines[i][:-1].rstrip()
                    lines[i] = lines[i] + ", LLM\n"
                else:
                    lines[i] = ln.rstrip("\n").rstrip() + ", LLM\n"
            return "".join(lines)
    # No vllm import line found; prepend a simple one
    return "from vllm import LLM\n" + text


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

        # 3) Remove build_embed_engine import and inline LLM-based embed engine construction
        # Remove: from training_free_grpo.semiconductor.embeddings import build_model as build_embed_engine
        before = text
        text = text.replace(
            "from training_free_grpo.semiconductor.embeddings import build_model as build_embed_engine\n",
            "",
        )
        # Replace: embed_engine = build_embed_engine(EMBED_MODEL_ID...) -> LLM(..., task='embed')
        text = re.sub(
            r"embed_engine\s*=\s*build_embed_engine\s*\(\s*EMBED_MODEL_ID[^\)]*\)",
            (
                "embed_engine = LLM(\n"
                "    model=EMBED_MODEL_ID,\n"
                "    task='embed',\n"
                "    trust_remote_code=True,\n"
                "    tensor_parallel_size=TENSOR_PARALLEL_SIZE,\n"
                "    dtype=DTYPE,\n"
                "    gpu_memory_utilization=0.1,\n"
                ")"
            ),
            text,
        )
        # Also patch existing direct LLM embed_engine instantiation to include explicit args
        text = re.sub(
            r"embed_engine\s*=\s*LLM\(\s*model=EMBED_MODEL_ID,\s*task='embed',\s*trust_remote_code=True\s*\)",
            (
                "embed_engine = LLM(\n"
                "    model=EMBED_MODEL_ID,\n"
                "    task='embed',\n"
                "    trust_remote_code=True,\n"
                "    tensor_parallel_size=TENSOR_PARALLEL_SIZE,\n"
                "    dtype=DTYPE,\n"
                "    gpu_memory_utilization=0.1,\n"
                ")"
            ),
            text,
        )
        if text != before:
            # Ensure LLM import exists in this cell when we changed it
            text = ensure_llm_import(text)

        # 4) Sanitize config: remove env-based globals and set explicit literals
        # Paths
        text = text.replace(
            'MODEL_PATH = os.environ.get("VLLM_MODEL", "/mnt/storage/models/Qwen3/Qwen3-32B")',
            'MODEL_PATH = "/mnt/storage/models/Qwen3/Qwen3-32B"',
        )
        text = text.replace(
            'GRADING_MODEL_PATH = os.environ.get("VLLM_MODEL_GRADING", "/mnt/storage/models/Qwen3/Qwen3-Next-80B-A3B-Thinking")',
            'GRADING_MODEL_PATH = "/mnt/storage/models/Qwen3/Qwen3-Next-80B-A3B-Thinking"',
        )
        text = text.replace(
            'DATASET_PATH = os.environ.get("SEMI_DATASET", "/mnt/workspace/MLLM/zz/training_free_grpo/semiconductor/data/3.5k_filtered_processed_data.json")',
            'DATASET_PATH = "/mnt/workspace/MLLM/zz/training_free_grpo/semiconductor/data/3.5k_filtered_processed_data.json"',
        )
        text = text.replace(
            'EXPERIMENT_DIR = os.environ.get("SEMI_EXP_DIR", "/mnt/workspace/MLLM/zz/training_free_grpo/semiconductor")',
            'EXPERIMENT_DIR = "/mnt/workspace/MLLM/zz/training_free_grpo/semiconductor"',
        )
        # Randomization and engine configs
        text = text.replace('RANDOM_SEED = int(os.environ.get("SEED", "42"))', 'RANDOM_SEED = 42')
        text = text.replace('SHUFFLE_EACH_STEP = os.environ.get("SHUFFLE_EACH_STEP", "True").lower() in ["1", "true", "yes"]', 'SHUFFLE_EACH_STEP = True')
        text = text.replace('TENSOR_PARALLEL_SIZE = int(os.environ.get("TP_SIZE", "8"))', 'TENSOR_PARALLEL_SIZE = 8')
        text = text.replace('MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))', 'MAX_MODEL_LEN = 32768')
        text = text.replace('DTYPE = os.environ.get("DTYPE", "auto")  # "auto", "float16", "bfloat16"', 'DTYPE = "auto"  # "auto", "float16", "bfloat16"')
        text = text.replace('GRADING_TP_SIZE = int(os.environ.get("GRADING_TP_SIZE", str(TENSOR_PARALLEL_SIZE)))', 'GRADING_TP_SIZE = TENSOR_PARALLEL_SIZE')
        text = text.replace('GRADING_MAX_MODEL_LEN = int(os.environ.get("GRADING_MAX_MODEL_LEN", str(MAX_MODEL_LEN)))', 'GRADING_MAX_MODEL_LEN = MAX_MODEL_LEN')
        text = text.replace('GRADING_DTYPE = os.environ.get("GRADING_DTYPE", DTYPE)', 'GRADING_DTYPE = DTYPE')
        # Experiences and retrieval
        text = text.replace('EXPERIENCES_BASE = os.environ.get("EXPERIENCES_ROOT", os.path.join(EXPERIMENT_DIR, "experiences"))', 'EXPERIENCES_BASE = os.path.join(EXPERIMENT_DIR, "experiences")')
        text = text.replace('TOP_K_EXPERIENCES = int(os.environ.get("TOP_K_EXPERIENCES", "5"))', 'TOP_K_EXPERIENCES = 5')
        text = text.replace('EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "Qwen/Qwen3-Embedding-8B")', 'EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"')
        text = text.replace('EMBED_DEVICE = os.environ.get("EMBED_DEVICE")  # e.g., "cuda" or None', 'EMBED_DEVICE = None  # e.g., "cuda" or None')
        text = text.replace('EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "256"))', 'EMBED_BATCH_SIZE = 256')
        text = text.replace('EMBED_USE_FA2 = os.environ.get("EMBED_USE_FA2", "false").lower() in ["1", "true", "yes"]', 'EMBED_USE_FA2 = False')
        text = text.replace('EMBED_INSTRUCTION = os.environ.get("EMBED_INSTRUCTION", "").strip() or None', 'EMBED_INSTRUCTION = None')
        text = text.replace('EXPERIENCES_INDEX_DIR = os.environ.get("EXPERIENCES_INDEX_DIR", os.path.join(EXPERIENCES_BASE, "index"))', 'EXPERIENCES_INDEX_DIR = os.path.join(EXPERIENCES_BASE, "index")')
        text = text.replace('EMBED_REBUILD_INDEX = os.environ.get("EMBED_REBUILD_INDEX", "false").lower() in ["1", "true", "yes"]', 'EMBED_REBUILD_INDEX = False')
        # Remove CUDA_VISIBLE_DEVICES setting
        text = text.replace('os.environ["CUDA_VISIBLE_DEVICES"] = \'0, 1, 2, 3, 4, 5, 6, 7\'', '')

        cell["source"] = text.splitlines(True)

    Path(NB_PATH).write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Notebook simplified: removed unnecessary try/except blocks.")


if __name__ == "__main__":
    main()
