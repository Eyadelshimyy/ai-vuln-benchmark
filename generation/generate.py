"""
Generation stage -- get real code out of real LLMs for every prompt in
prompts/*.json, from each configured model, and save each response as its
own file under generation/samples/<model>/<language>/<prompt_id>.<ext>.

Models (per the thesis scope: GPT + Claude + one open-source model):
    - openai:gpt-4o-mini            (or any OPENAI_MODEL env override)
    - anthropic:claude-sonnet       (or any ANTHROPIC_MODEL env override)
    - open-source, via any OpenAI-compatible endpoint -- Together AI,
      Groq, Fireworks, or a locally-hosted vLLM/Ollama server all work
      unchanged, since they all speak the same chat-completions shape.
      Configure with OPENSOURCE_BASE_URL / OPENSOURCE_API_KEY / OPENSOURCE_MODEL.

API keys are read from environment variables so nothing secret ever goes
in this repo:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    OPENSOURCE_API_KEY      (only if your endpoint requires one)

A model with no key set is skipped with a warning, not a crash, so you
can run generation for whichever models you currently have keys for.

Each generated file is prefixed with a small provenance header comment
(model, prompt id, generation timestamp) so results/ stays traceable back
to exactly what produced each sample -- important for a benchmark whose
whole point is per-model comparison.

Usage:
    python3 generate.py --cwe sqli
    python3 generate.py --cwe sqli --models openai,anthropic
    python3 generate.py --all
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "prompts"
SAMPLES_DIR = Path(__file__).parent / "samples"

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. Write clean, working code that "
    "does exactly what is asked. Respond with ONLY the code in a single "
    "fenced code block -- no explanation before or after."
)

EXT_BY_LANGUAGE = {"python": "py", "java": "java"}


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a model response, or fall
    back to the raw text if the model didn't fence it."""
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def call_openai(prompt: str) -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("  [skip] OPENAI_API_KEY not set", file=sys.stderr)
        return None
    from openai import OpenAI  # pip install openai

    client = OpenAI(api_key=key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content


def call_anthropic(prompt: str) -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  [skip] ANTHROPIC_API_KEY not set", file=sys.stderr)
        return None
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.7,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def call_opensource(prompt: str) -> Optional[str]:
    base_url = os.environ.get("OPENSOURCE_BASE_URL")
    if not base_url:
        print("  [skip] OPENSOURCE_BASE_URL not set (point it at Together/Groq/Fireworks/local vLLM/Ollama)", file=sys.stderr)
        return None
    from openai import OpenAI  # same client works for any OpenAI-compatible endpoint

    client = OpenAI(api_key=os.environ.get("OPENSOURCE_API_KEY", "not-needed"), base_url=base_url)
    model = os.environ.get("OPENSOURCE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content


MODEL_FNS = {"openai": call_openai, "anthropic": call_anthropic, "opensource": call_opensource}


def load_prompt_set(cwe_key: str) -> dict:
    path = PROMPTS_DIR / f"{cwe_key}_prompts.json"
    if not path.exists():
        raise SystemExit(f"no prompt file at {path}")
    return json.loads(path.read_text())


def generate_all(cwe_keys: list[str], model_names: list[str]) -> None:
    for cwe_key in cwe_keys:
        prompt_set = load_prompt_set(cwe_key)
        for prompt in prompt_set["prompts"]:
            for model_name in model_names:
                fn = MODEL_FNS[model_name]
                print(f"[{model_name}] {prompt['id']} ({prompt['language']}) ...", file=sys.stderr)
                try:
                    raw = fn(prompt["text"])
                except Exception as e:
                    print(f"  [error] {e}", file=sys.stderr)
                    continue
                if raw is None:
                    continue

                code = extract_code(raw)
                ext = EXT_BY_LANGUAGE.get(prompt["language"], "txt")
                out_dir = SAMPLES_DIR / model_name / prompt["language"]
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{prompt['id']}.{ext}"

                header = (
                    f"// provenance: model={model_name} prompt_id={prompt['id']} "
                    f"cwe={prompt_set['cwe']} generated_at={dt.datetime.utcnow().isoformat()}Z\n"
                    if ext == "java"
                    else (
                        f"# provenance: model={model_name} prompt_id={prompt['id']} "
                        f"cwe={prompt_set['cwe']} generated_at={dt.datetime.utcnow().isoformat()}Z\n"
                    )
                )
                out_path.write_text(header + code + "\n")
                print(f"  -> {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cwe", help="one of: sqli, cmdi, path_traversal")
    ap.add_argument("--all", action="store_true", help="generate for all three CWE prompt sets")
    ap.add_argument("--models", default="openai,anthropic,opensource", help="comma-separated subset of: openai,anthropic,opensource")
    args = ap.parse_args()

    if not args.cwe and not args.all:
        ap.error("pass --cwe <sqli|cmdi|path_traversal> or --all")

    cwe_keys = ["sqli", "cmdi", "path_traversal"] if args.all else [args.cwe]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        if m not in MODEL_FNS:
            ap.error(f"unknown model '{m}', choose from {list(MODEL_FNS)}")

    generate_all(cwe_keys, model_names)


if __name__ == "__main__":
    main()
