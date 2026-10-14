"""
Build and refresh the local models database payload.

Outputs:
- core/data/models_db.json
Optional:
- imports the JSON payload into core/data/models.db via SQLite
"""
import argparse
import datetime
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(ROOT, "core", "data", "models_db.json")


def collect_models_data():
    models_data = {}

    kijai_path = os.path.join(ROOT, "data", "samples", "kijai_all_models.txt")
    if os.path.exists(kijai_path):
        with open(kijai_path, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                filename = os.path.basename(path)
                key = filename.lower()
                models_data[key] = {
                    "repo_id": "Kijai/WanVideo_comfy",
                    "path": path,
                    "filename": filename,
                    "source": "Kijai",
                }

    comfy_path = os.path.join(ROOT, "data", "samples", "comfy_gguf_models.json")
    if os.path.exists(comfy_path):
        with open(comfy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for repo_id, files in data.items():
                source_type = "Comfy-Org" if "Comfy-Org" in repo_id else "GGUF"
                for path in files:
                    filename = os.path.basename(path)
                    key = filename.lower()
                    if key not in models_data:
                        models_data[key] = {
                            "repo_id": repo_id,
                            "path": path,
                            "filename": filename,
                            "source": source_type,
                        }

    return models_data


def write_json(models_data, output_file=OUTPUT_FILE):
    db_payload = {
        "meta": {
            "total_count": len(models_data),
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "CIVITAI_MAP": {
            "aniwan2114bfp8e4m3fn_i2v480pnew": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
            "aniwan2114bfp8e4m3fn": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
            "aniwan21t2v14b": "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors",
            "aniwani2v14b": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
            "wan_2_1_t2v_14b_rcm_lora_average_rank_83": "LoRAs/rCM/Wan_2_1_T2V_14B_480p_rCM_lora_average_rank_83_bf16.safetensors",
            "wan_2_1_t2v_14b_rcm_lora_average_rank_148": "LoRAs/rCM/Wan_2_1_T2V_14B_480p_rCM_lora_average_rank_148_bf16.safetensors",
            "wan_2_1_t2v_14b_720p_rcm": "LoRAs/rCM/Wan_2_1_T2V_14B_720p_rCM_lora_average_rank_94_bf16.safetensors",
            "infinitetalk_single": "InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors",
            "infinitetalk_multi": "InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp16.safetensors",
        },
        "MODELS_DB": models_data,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(db_payload, f, indent=2, ensure_ascii=False)

    print(f"Database JSON generated with {len(models_data)} entries at {output_file}")


def import_to_sqlite(json_path=OUTPUT_FILE):
    try:
        import sys
        # Ensure project root + core/ are importable so database.py can resolve
        # its `from config import get_matcher_config` fallback (and `from .config`).
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        core_dir = os.path.join(ROOT, "core")
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)

        db_path = os.path.join(ROOT, "core", "data", "models.db")
        spec = importlib.util.spec_from_file_location(
            "database_mod", os.path.join(ROOT, "core", "database.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["database_mod"] = mod
        spec.loader.exec_module(mod)
        # Ensure schema exists (migrates older DBs automatically), then wipe the
        # external_models table so the import fully replaces any stale rows.
        # The stale 2026-02-01 db lacked this table entirely, which silently
        # disabled the matcher's DB-first strategy.
        db = mod.ModelDatabase(db_path)
        conn = db._get_connection()
        conn.execute("DELETE FROM external_models")
        conn.commit()
        conn.close()
        count = db.import_models_db_json(json_path)
        print(f"SQLite import complete: {count} records")
        return count
    except Exception as e:
        print(f"SQLite import failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(description="Build and refresh the models DB payload")
    parser.add_argument("--import-sqlite", action="store_true", help="Import the generated JSON into SQLite")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Path to the JSON output file")
    args = parser.parse_args()

    models_data = collect_models_data()
    write_json(models_data, output_file=args.output)
    if args.import_sqlite:
        import_to_sqlite(args.output)


if __name__ == "__main__":
    main()
