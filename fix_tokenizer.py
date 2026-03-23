"""
fix_tokenizer.py
================
Rebuilds the local donut/weights tokenizer files so they are compatible with
the installed `tokenizers 0.14.x` library that ships with `transformers 4.35`.

Run this script ONCE after cloning the repo or after fine-tuning on Colab
(where a newer tokenizers version may have been used to save the files).

Strategy
--------
1. Extract the custom special tokens from the existing (incompatible)
   tokenizer.json by reading its added_tokens list.
2. Load the base donut-base tokenizer from HuggingFace Hub (always compatible
   with tokenizers 0.14).
3. Add the custom tokens back.
4. Save to donut/weights, overwriting tokenizer.json + tokenizer_config.json.
5. Patch config.json's decoder_start_token_id to match <s_general> in the
   rebuilt vocab.
"""

import os, sys, json
from transformers import AutoTokenizer, PreTrainedTokenizerFast

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "donut", "weights")

# Known custom tokens from the original fine-tuned tokenizer_config.json.
# Used as a fallback when the tokenizer.json has already been overwritten
# and no longer contains these entries.
KNOWN_CUSTOM_TOKENS = [
    # Task-start tokens
    "<s_general>", "<s_lot_geometry>", "<s_parcel_info>", "<s_tabular_data>",
    # Schema open tags
    "<adj_id>", "<arc>", "<area_val>", "<az>", "<chord>", "<dist>", "<east>",
    "<lot_id>", "<north>", "<plan_title>", "<pt_id>", "<radius>", "<row>",
    "<street>", "<text>",
    # Schema close tags
    "</adj_id>", "</arc>", "</area_val>", "</az>", "</chord>", "</dist>",
    "</east>", "</lot_id>", "</north>", "</plan_title>", "</pt_id>",
    "</radius>", "</row>", "</street>", "</text>",
]

BASE_MODEL  = "naver-clova-ix/donut-base"

# ── IDs of base donut-base special tokens (do not re-add these) ──────────────
# <s>=0, <pad>=1, </s>=2, <unk>=3, <mask>=57521, <sep/>=57522,
# <s_iitcdip>=57523, <s_synthdog>=57524
BASE_SPECIAL_IDS = {0, 1, 2, 3, 57521, 57522, 57523, 57524}


def extract_custom_tokens_from_json(tokenizer_json_path: str) -> list[str]:
    """
    Read added_tokens from tokenizer.json and return those that are NOT part
    of the base donut-base vocabulary (i.e., the fine-tuned custom tokens).
    """
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    custom = []
    for entry in data.get("added_tokens", []):
        if entry["id"] not in BASE_SPECIAL_IDS:
            custom.append(entry["content"])
    return custom


def main():
    tokenizer_json = os.path.join(WEIGHTS_DIR, "tokenizer.json")
    config_json    = os.path.join(WEIGHTS_DIR, "config.json")
    tokenizer_cfg  = os.path.join(WEIGHTS_DIR, "tokenizer_config.json")

    # ── Step 1: Extract custom tokens from the existing (incompatible) file ──
    print("[1/5] Extracting custom tokens from existing tokenizer.json ...")
    if not os.path.isfile(tokenizer_json):
        print("ERROR: donut/weights/tokenizer.json not found.", file=sys.stderr)
        sys.exit(1)

    # Try to parse it with the current library; if it fails, fall back to
    # raw JSON extraction (which always works regardless of tokenizers version).
    try:
        import json as _json
        custom_tokens = extract_custom_tokens_from_json(tokenizer_json)
        print(f"       Extracted {len(custom_tokens)} custom tokens via raw JSON.")
    except Exception as e:
        print(f"       Raw JSON extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not custom_tokens:
        print("       No custom tokens in tokenizer.json (may have been overwritten).")
        print(f"       Falling back to hardcoded KNOWN_CUSTOM_TOKENS ({len(KNOWN_CUSTOM_TOKENS)} tokens).")
        custom_tokens = KNOWN_CUSTOM_TOKENS
    else:
        print(f"       First 5: {custom_tokens[:5]}")


    # ── Step 2: Load base tokenizer from Hub ─────────────────────────────────
    print("[2/5] Loading base tokenizer from HuggingFace Hub ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    print(f"       Base vocab size: {len(tok)}")

    # ── Step 3: Add custom tokens ─────────────────────────────────────────────
    print("[3/5] Adding custom schema tokens ...")
    if custom_tokens:
        # Separate task-start tokens (<s_*>) from schema tags (<tag>, </tag>)
        task_tokens  = [t for t in custom_tokens if t.startswith("<s_")]
        schema_tokens = [t for t in custom_tokens if not t.startswith("<s_")]
        tok.add_special_tokens({"additional_special_tokens": task_tokens + schema_tokens})
        print(f"       Final vocab size: {len(tok)}")
    else:
        print("       Nothing to add.")

    sg_id = tok.convert_tokens_to_ids("<s_general>")
    print(f"       <s_general> id: {sg_id}")

    # ── Step 4: Save the rebuilt tokenizer ────────────────────────────────────
    print("[4/5] Saving updated tokenizer to donut/weights ...")
    tok.save_pretrained(WEIGHTS_DIR)

    # Patch tokenizer_config.json: remove tokenizer_class (causes sentencepiece lookup)
    if os.path.isfile(tokenizer_cfg):
        with open(tokenizer_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        removed = []
        for key in ("tokenizer_class", "auto_map", "from_slow", "is_local"):
            if key in cfg:
                cfg.pop(key)
                removed.append(key)
        # Persist extra_special_tokens list for documentation / future re-runs
        cfg["extra_special_tokens"] = custom_tokens
        with open(tokenizer_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"       Removed from tokenizer_config.json: {removed}")
        print(f"       Saved extra_special_tokens list ({len(custom_tokens)} tokens).")

    # Patch config.json decoder_start_token_id
    if os.path.isfile(config_json) and sg_id is not None:
        with open(config_json, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        old_id = model_cfg.get("decoder_start_token_id")
        if old_id != sg_id:
            model_cfg["decoder_start_token_id"] = sg_id
            with open(config_json, "w", encoding="utf-8") as f:
                json.dump(model_cfg, f, indent=2)
            print(f"       config.json: decoder_start_token_id {old_id} -> {sg_id}")
        else:
            print(f"       config.json decoder_start_token_id already {sg_id}.")

    # ── Step 5: Verify reload ────────────────────────────────────────────────
    print("[5/5] Verifying reload from local tokenizer.json ...")
    try:
        tok2 = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
        print(f"       Reload OK. Vocab size: {len(tok2)}")
        sg2 = tok2.convert_tokens_to_ids("<s_general>")
        print(f"       <s_general> id in reloaded tokenizer: {sg2}")
        if sg2 is None:
            print("       WARNING: <s_general> not found in reloaded tokenizer.")
    except Exception as e:
        print(f"       Reload FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone! The tokenizer files in donut/weights are now compatible.")
    print("Please restart your FastAPI backend to pick up the changes.")


if __name__ == "__main__":
    main()
