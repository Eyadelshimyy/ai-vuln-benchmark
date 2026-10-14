"""AI-assisted "is this job open to a Canada-based candidate?" classifier.

The regex-based heuristics in generate_new_jobs.py / generate_companies_page.py
are a reasonable first pass but are finicky with the huge variety of
freeform location strings ATSs use ("CA Remote", "Americas", "Hybrid -
Toronto or Remote (US)", etc). This module asks GitHub Copilot CLI
(https://docs.github.com/en/copilot/how-tos/copilot-cli/automate-copilot-cli/automate-with-actions)
to classify each *distinct* location string instead, and caches the result
on disk so repeat locations across crawl runs never need a second AI call.

Designed to fail soft: if the `copilot` CLI isn't installed/authenticated
(e.g. a local dev machine, or the COPILOT_GITHUB_TOKEN secret isn't
configured), classify() returns an empty mapping and callers should keep
using their existing regex-based result for every location.

Disable entirely (e.g. for a quick local test) by setting
DISABLE_AI_CANADA=1 in the environment.
"""

import json
import os
import shutil
import subprocess
import tempfile

CACHE_PATH = os.path.join("state", "canada_ai_cache.json")

_PROMPT_TEMPLATE = """You are classifying job location strings for a job board that only cares about one thing: could a candidate who is physically based in Canada realistically apply to and be hired for this specific posting?

Read the JSON array of distinct location strings from the file at {input_path}. Each string is copied verbatim from a real job posting (city/region names, remote-work wording, etc).

For EVERY string in that array, decide true or false:
- true: the location is a Canadian city/province, says "Canada" outright, is remote/worldwide/North-America-wide wording that a Canada-based person could reasonably apply under (e.g. "Remote", "Remote - Global", "North America", "Americas", "Remote (US or Canada)"), or otherwise doesn't rule out Canada.
- false: the location clearly restricts the role to a specific non-Canadian place with no remote/global option (e.g. "San Francisco, CA", "London, UK", "Remote (US only)", "APAC").

Write a JSON object to the file at {output_path} mapping EACH input string (used verbatim as the key) to its boolean classification. Do not add, omit, or reword any keys. Do not write any other output. Do not ask me any questions.
"""


def _load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _run_copilot_cli(locations):
    """Classify `locations` (a list of distinct strings) via Copilot CLI.

    Returns a dict mapping location -> bool for whatever Copilot
    successfully classified, or {} if the CLI isn't available / the call
    failed / the response couldn't be parsed. Never raises.
    """
    if not locations:
        return {}
    if shutil.which("copilot") is None:
        print("  [ai_canada] 'copilot' CLI not found on PATH; skipping AI classification")
        return {}

    tmp_dir = tempfile.mkdtemp(prefix="ai_canada_")
    input_path = os.path.join(tmp_dir, "locations.json")
    output_path = os.path.join(tmp_dir, "classifications.json")
    try:
        with open(input_path, "w") as f:
            json.dump(locations, f, ensure_ascii=False)

        prompt = _PROMPT_TEMPLATE.format(input_path=input_path, output_path=output_path)
        result = subprocess.run(
            [
                "copilot", "-p", prompt,
                "--allow-tool=read", "--allow-tool=write",
                "--no-ask-user",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  [ai_canada] copilot CLI exited {result.returncode}: {result.stderr[-500:]}")
            return {}

        with open(output_path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            print("  [ai_canada] copilot CLI output was not a JSON object; ignoring")
            return {}

        return {
            loc: bool(raw[loc])
            for loc in locations
            if loc in raw and isinstance(raw[loc], bool)
        }
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  [ai_canada] classification failed, falling back to heuristic: {e}")
        return {}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def classify(locations):
    """Return {location: is_canada} for every string in `locations` that
    the AI (cached or freshly classified) has an opinion on. Strings it
    couldn't classify are simply absent from the result — callers should
    keep their regex-based fallback for those.
    """
    if os.environ.get("DISABLE_AI_CANADA"):
        return {}

    distinct = sorted({loc for loc in locations if loc})
    if not distinct:
        return {}

    cache = _load_cache()
    uncached = [loc for loc in distinct if loc not in cache]

    if uncached:
        print(f"  [ai_canada] classifying {len(uncached)} new location string(s) via Copilot CLI")
        fresh = _run_copilot_cli(uncached)
        if fresh:
            cache.update(fresh)
            _save_cache(cache)

    return {loc: cache[loc] for loc in distinct if loc in cache}
