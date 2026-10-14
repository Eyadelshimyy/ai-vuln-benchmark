#!/usr/bin/env python3
"""Check, resolve, and migrate the saved state in <repository>/.nthplusio-skills/.

  state.py check [discovery|orchestration] [--root DIR]
  state.py resolve --harness NAME [--orchestration] [--root DIR]
  state.py migrate [--from PATH] [--dry-run] [--root DIR]

check    Validates discovery.json and orchestration.json, or only the one named.
resolve  Prints one harness's discovery entry. NAME is its saved key or that
         key's slug. --orchestration adds the harness's orchestration.json
         profile, which orchestration-run consumes.
migrate  Moves state saved by earlier versions of these skills into
         .nthplusio-skills/, upgrades it to the current schema, checks it, and
         prints a JSON summary. --from names an earlier configuration outside
         the repository's legacy folders; it is copied, not moved.

No command invokes harness tools. --root defaults to the Git repository that
holds the working directory.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess


STATE_DIR = ".nthplusio-skills"
FILES = {"discovery": "discovery.json", "orchestration": "orchestration.json"}
CURRENT = {"discovery": 1, "orchestration": 3}
OWNERS = {"discovery": "harness-discovery", "orchestration": "orchestration-setup"}
# A failure that already says what to run must not gain a second skill name.
HINTS = (
    "run harness-discovery",
    "run orchestration-setup",
    "run state.py migrate",
    "repair this harness profile with orchestration-setup",
)
active = "discovery"


def use(kind):
    global active
    active = kind


def with_owner(message):
    """Append the skill that owns the file being checked, unless a hint is already present."""
    text = str(message)
    if any(hint in text for hint in HINTS):
        return text
    return f"{text}; run {OWNERS[active]}"


def load_saved(path, kind):
    use(kind)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        raise ValueError(f"{path.name} is not valid JSON")


# Earlier homes of the saved state, oldest first. Each held config.json and the
# discovery file its discovery_file named, by default discovery.json.
LEGACY_DIRS = (".orchestration",)
COORDINATOR_SKILLS = {"harness-discovery", "orchestration-setup", "orchestration-run"}
# Business stage keys from orchestration-run's dashboard data contract.
STAGES = ("defined", "researched", "pr_readiness", "code_publication", "review", "verification",
          "deployment", "business_acceptance")


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def texts(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of text")
    for item in value:
        text(item, label)


def slug(name):
    """`Claude Code` -> `claude-code`: the name callers use for files and flags."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def check_version(document, kind):
    use(kind)
    found = document["schema_version"]
    if found == CURRENT[kind]:
        return
    if type(found) is int and found < CURRENT[kind]:
        raise ValueError(f"{kind} schema_version {found} is older than {CURRENT[kind]}; run state.py migrate")
    raise ValueError(f"unsupported {kind} schema_version {found!r}; update these skills")


def check_discovery(discovery):
    use("discovery")
    check_version(discovery, "discovery")
    repository = discovery["repository"]
    for field in ("guidance", "ticket_sources", "notes"):
        texts(repository[field], f"repository.{field}")
    if repository["ticket_template"] is not None:
        text(repository["ticket_template"], "repository.ticket_template")
    if not isinstance(discovery["harnesses"], dict) or not discovery["harnesses"]:
        raise ValueError("harnesses must contain a discovered harness name")
    slugs = {}
    for name, facts in discovery["harnesses"].items():
        text(name, "harness name")
        key = slug(name)
        if not key:
            raise ValueError(f"{name}: a harness name needs letters or digits")
        if key in slugs:
            raise ValueError(f"{slugs[key]} and {name} share the slug {key}; harness names must differ in letters or digits")
        slugs[key] = name
        for field in ("context", "owner_term", "helper_term"):
            text(facts[field], f"{name}.{field}")
        observed = datetime.fromisoformat(facts["observed_at"].replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError("observed_at needs a timezone")
        texts(facts["sources"], f"{name}.sources")
        if not facts["sources"]:
            raise ValueError("sources must identify the inspected tools or documentation")
        for section, fields in {
            "owners": ("launch", "inspect", "message", "collect", "close", "resume"),
            "helpers": ("invoke", "collect"),
            "monitoring": ("notifications", "wait", "wake"),
            "skills": ("load",),
            "evidence": ("transfer", "retain_before_close"),
        }.items():
            for field in fields:
                if facts[section][field] is not None:
                    text(facts[section][field], f"{name}.{section}.{field}")
        if facts["owners"]["workspace_model"] not in ("shared", "isolated", "selectable", "unknown"):
            raise ValueError("owners.workspace_model must describe workspace sharing")
        for field in ("blocks_caller", "follow_up"):
            if facts["helpers"][field] is not None and type(facts["helpers"][field]) is not bool:
                raise ValueError(f"helpers.{field} must be boolean or null")
        text(facts["skills"]["brief"], "skills.brief")
        for section, fields in {"helpers": ("readonly",), "skills": ("locations", "reload"),
                                "instructions": ("user", "project"), "models": ("select", "list")}.items():
            if section not in facts:
                continue
            if not isinstance(facts[section], dict):
                raise ValueError(f"{name}.{section} must be an object")
            for field in fields:
                if facts[section].get(field) is not None:
                    text(facts[section][field], f"{name}.{section}.{field}")
        if facts.get("ask_user") is not None:
            text(facts["ask_user"], f"{name}.ask_user")
        texts(facts["limitations"], "limitations")
        if not isinstance(facts["artifacts"], list):
            raise ValueError("artifacts must be a list")
        seen = set()
        for artifact in facts["artifacts"]:
            for field in ("id", "label", "present", "update", "audience"):
                text(artifact[field], f"artifact.{field}")
            if artifact["id"] in seen:
                raise ValueError("artifact IDs must be unique per harness")
            seen.add(artifact["id"])
            for field in ("interaction", "auto_refresh"):
                if artifact.get(field) is not None:
                    text(artifact[field], f"artifact.{field}")
    if "REPLACE_WITH_" in json.dumps(discovery):
        raise ValueError("replace every template placeholder with a discovered value")


def check_orchestration(config, discovery):
    use("orchestration")
    check_version(config, "orchestration")
    check_discovery(discovery)
    use("orchestration")
    policy = config["project"]["ticket_policy"]
    for field in ("source", "location", "definition", "template", "id_convention", "confirmed_from"):
        text(policy[field], f"ticket_policy.{field}")
    if type(policy["criteria_limit"]) is not int or policy["criteria_limit"] < 1:
        raise ValueError("ticket_policy.criteria_limit must be a positive integer")

    def shared(values):
        texts(values, "shared_skills")
        if COORDINATOR_SKILLS & set(values):
            raise ValueError("harness-discovery, orchestration-setup and orchestration-run belong to the coordinator, not shared workers")

    shared(config["project"]["shared_skills"])
    stage_skills = config["project"].get("stage_skills", {})
    if not isinstance(stage_skills, dict) or set(stage_skills) - set(STAGES):
        raise ValueError(f"project.stage_skills keys must be business stages: {', '.join(STAGES)}")
    for values in stage_skills.values():
        shared(values)
    if not isinstance(config["harnesses"], dict) or not config["harnesses"]:
        raise ValueError("harnesses must contain a confirmed setup profile")
    for name, profile in config["harnesses"].items():
        if name not in discovery["harnesses"]:
            raise ValueError(f"{name}: no saved discovery; run harness-discovery")
        text(profile["run_root"], "run_root")
        shared(profile["shared_skills"])
        text(profile["confirmed_from"], "confirmed_from")
        defaults = profile["defaults"]
        text(defaults["finish_line"], "defaults.finish_line")
        if type(defaults["interval_seconds"]) is not int or defaults["interval_seconds"] < 1:
            raise ValueError("defaults.interval_seconds must be a positive integer")
        destination = defaults["destination"]
        if destination is not None and destination not in {item["id"] for item in discovery["harnesses"][name]["artifacts"]}:
            raise ValueError(f"{name}: default destination must be a discovered artifact ID or null")
    if "REPLACE_WITH_" in json.dumps(config):
        raise ValueError("replace every template placeholder with a confirmed value")


def repository_root(given):
    if given:
        return Path(given).expanduser().resolve()
    found = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if found.returncode:
        raise ValueError("not inside a Git repository; pass --root")
    return Path(found.stdout.strip()).resolve()


def state_path(root, kind):
    return root / STATE_DIR / FILES[kind]


def legacy_files(root):
    return [path for name in LEGACY_DIRS for path in (root / name / "config.json", root / name / "discovery.json")
            if path.is_file()]


def saved(root, kind):
    """The current file and its document, or an error naming what to run."""
    use(kind)
    path = state_path(root, kind)
    if path.is_file():
        return path, load_saved(path, kind)
    legacy = legacy_files(root)
    if legacy:
        raise ValueError(f"legacy state at {legacy[0]}; run state.py migrate")
    raise ValueError(f"no {FILES[kind]} in {path.parent}; run {OWNERS[kind]}")


def check(root, scope=None):
    kinds = [scope] if scope else [kind for kind in FILES if state_path(root, kind).is_file()] or ["discovery"]
    discovery_path, discovery = saved(root, "discovery")
    check_discovery(discovery)
    lines = [f"PASS: discovery {discovery_path}"] if "discovery" in kinds else []
    if "orchestration" in kinds:
        config_path, config = saved(root, "orchestration")
        check_orchestration(config, discovery)
        lines.append(f"PASS: orchestration {config_path}")
    return lines


def match(harnesses, name):
    if name in harnesses:
        return name
    found = [key for key in harnesses if slug(key) == slug(name)]
    if len(found) != 1:
        raise ValueError(f"{name}: no saved discovery for this harness; run harness-discovery in it")
    return found[0]


def resolve(root, name, orchestration=False):
    _, discovery = saved(root, "discovery")
    check_discovery(discovery)
    key = match(discovery["harnesses"], name)
    facts = deepcopy(discovery["harnesses"][key])
    facts["harness"] = {"name": key, "slug": slug(key), "context": facts["context"],
                        "owner_term": facts["owner_term"], "helper_term": facts["helper_term"]}
    if not orchestration:
        return facts
    config_path, config = saved(root, "orchestration")
    check_orchestration(config, discovery)
    if key not in config["harnesses"]:
        raise ValueError(f"{key}: no setup profile; run orchestration-setup for this harness")
    profile = config["harnesses"][key]
    run_root = Path(profile["run_root"]).expanduser()
    if PureWindowsPath(profile["run_root"]).is_absolute() and not run_root.is_absolute():
        raise ValueError("run_root is for another operating system; repair this harness profile with orchestration-setup")
    facts.update(config_path=str(config_path), run_root=str((config_path.parent / run_root).resolve()),
                 ticket_policy=deepcopy(config["project"]["ticket_policy"]), defaults=deepcopy(profile["defaults"]))
    facts["skills"]["shared"] = list(dict.fromkeys(config["project"]["shared_skills"] + profile["shared_skills"]))
    facts["skills"]["stages"] = deepcopy(config["project"].get("stage_skills", {}))
    return facts


class Relocation:
    """Keeps a relative path inside a moved file pointing at the same place."""

    def __init__(self, root, source_dir, target_dir):
        self.root, self.source_dir, self.target_dir = root, source_dir, target_dir
        self.rebased = []

    def path(self, value):
        if Path(value).expanduser().is_absolute() or PureWindowsPath(value).is_absolute():
            return value
        target = (self.source_dir / value).resolve()
        if target == self.root or self.root in target.parents:
            return os.path.relpath(target, self.target_dir)
        return str(target)


def orchestration_2_to_3(config, relocation):
    """discovery_file goes: discovery.json now always sits beside orchestration.json."""
    config = deepcopy(config)
    config.pop("discovery_file", None)
    for name, profile in config.get("harnesses", {}).items():
        before = profile.get("run_root")
        if isinstance(before, str) and before.strip():
            after = relocation.path(before)
            if after != before:
                profile["run_root"] = after
                relocation.rebased.append({"harness": name, "from": before, "to": after})
    config["schema_version"] = 3
    return config


# Each step upgrades a document from the version that keys it to the next one.
# A change to a saved file's shape adds a step here and a fixture of the old
# shape under harness-discovery/evals/fixtures/.
UPGRADES = {"discovery": {}, "orchestration": {2: orchestration_2_to_3}}


def upgrade(kind, document, relocation, path):
    use(kind)
    steps = []
    while document.get("schema_version") != CURRENT[kind]:
        found = document.get("schema_version")
        if found not in UPGRADES[kind]:
            raise ValueError(f"{path}: no upgrade from {kind} schema_version {found!r}; update these skills")
        document = UPGRADES[kind][found](document, relocation)
        steps.append({"file": str(path), "from": found, "to": document["schema_version"]})
    return document, steps


def tracked(root, path):
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(path)], capture_output=True)
    return listed.returncode == 0


def migrate(root, source=None, dry_run=False):
    state = root / STATE_DIR
    summary = {"state": str(state), "dry_run": dry_run, "moved": [], "copied": [], "removed_duplicates": [],
               "upgraded": [], "rebased_run_roots": [], "interview": []}
    found = {}
    config_source = Path(source).expanduser().resolve() if source else next(
        (path for path in legacy_files(root) if path.name == "config.json"), None)
    if config_source is not None:
        config = load_saved(config_source, "orchestration")
        if config.get("schema_version") == 1:
            summary["interview"].append(f"{config_source} is a version 1 configuration; "
                                        "convert it through the orchestration-setup interview")
        else:
            found["orchestration"] = (config_source, config)
            discovery_source = (config_source.parent / config.get("discovery_file", FILES["discovery"])).resolve()
            if not discovery_source.is_file():
                use("discovery")
                raise ValueError(f"{config_source} names {discovery_source}, which does not exist")
            found["discovery"] = (discovery_source, load_saved(discovery_source, "discovery"))
    if "discovery" not in found:
        legacy = next((path for path in legacy_files(root) if path.name == "discovery.json"), None)
        if legacy is not None:
            found["discovery"] = (legacy, load_saved(legacy, "discovery"))
    for kind in FILES:
        current = state_path(root, kind)
        if kind not in found and current.is_file():
            document = load_saved(current, kind)
            if document.get("schema_version") != CURRENT[kind]:
                found[kind] = (current, document)

    plans, final = {}, {}
    for kind, (path, document) in found.items():
        relocation = Relocation(root, path.parent, state)
        upgraded, steps = upgrade(kind, document, relocation, path)
        current = state_path(root, kind)
        if path != current and current.is_file():
            if load_saved(current, kind) != upgraded:
                raise ValueError(f"{path} and {current} both exist and differ; reconcile them, then run state.py migrate again")
            upgraded = None
        plans[kind] = (path, document, upgraded, steps, relocation.rebased)
    for kind in FILES:
        if kind in plans and plans[kind][2] is not None:
            final[kind] = plans[kind][2]
        elif state_path(root, kind).is_file():
            final[kind] = load_saved(state_path(root, kind), kind)
    if "discovery" in final:
        check_discovery(final["discovery"])
    if "orchestration" in final:
        if "discovery" not in final:
            raise ValueError("an orchestration configuration needs discovery beside it; run harness-discovery first")
        check_orchestration(final["orchestration"], final["discovery"])

    for kind, (path, original, upgraded, steps, rebased) in plans.items():
        current = state_path(root, kind)
        inside = root in path.parents
        if upgraded is not None:
            summary["upgraded"] += steps
            summary["rebased_run_roots"] += rebased
        if upgraded is None:
            if inside:
                summary["removed_duplicates"].append(str(path))
                if not dry_run and tracked(root, path):
                    subprocess.run(["git", "-C", str(root), "rm", "-q", "--", str(path)], check=True, capture_output=True)
                elif not dry_run:
                    path.unlink()
            continue
        content = path.read_bytes() if upgraded == original else (json.dumps(upgraded, indent=2) + "\n").encode()
        if path == current:
            if not dry_run:
                current.write_bytes(content)
            continue
        summary["moved" if inside else "copied"].append([str(path), str(current)])
        if dry_run:
            continue
        state.mkdir(exist_ok=True)
        if inside and tracked(root, path):
            subprocess.run(["git", "-C", str(root), "mv", "--", str(path), str(current)], check=True, capture_output=True)
        elif inside:
            path.rename(current)
        current.write_bytes(content)
    if not dry_run:
        for name in LEGACY_DIRS:
            folder = root / name
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        if any(state_path(root, kind).is_file() for kind in FILES):
            check(root)
    summary["changed"] = any(summary[key] for key in ("moved", "copied", "removed_duplicates", "upgraded"))
    return summary


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", help="repository holding .nthplusio-skills/; defaults to the working directory's Git repository")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", parents=[common], help="validate saved state")
    checking.add_argument("scope", nargs="?", choices=tuple(FILES))
    resolving = commands.add_parser("resolve", parents=[common], help="print one harness's saved facts")
    resolving.add_argument("--harness", required=True, help="saved harness key, or its slug")
    resolving.add_argument("--orchestration", action="store_true", help="add the harness's orchestration profile")
    migrating = commands.add_parser("migrate", parents=[common], help="move and upgrade state saved by earlier versions")
    migrating.add_argument("--from", dest="source", help="an earlier configuration outside the legacy folders")
    migrating.add_argument("--dry-run", action="store_true", help="report the changes without writing them")
    args = parser.parse_args()
    try:
        root = repository_root(args.root)
        if args.command == "check":
            print("\n".join(check(root, args.scope)))
        elif args.command == "resolve":
            print(json.dumps(resolve(root, args.harness, args.orchestration), indent=2))
        else:
            print(json.dumps(migrate(root, args.source, args.dry_run), indent=2))
    except KeyError as error:
        parser.exit(1, f"FAIL: {with_owner(f'missing field {error}')}\n")
    except (OSError, ValueError, TypeError, AttributeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FAIL: {with_owner(error)}\n")


if __name__ == "__main__":
    main()
