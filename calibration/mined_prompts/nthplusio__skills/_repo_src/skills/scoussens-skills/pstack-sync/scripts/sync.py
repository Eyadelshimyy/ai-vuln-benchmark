#!/usr/bin/env python3
"""Sync pstack skills from github.com/cursor/plugins (pstack/skills) into a skills directory.

  sync.py sync   --harness NAME [--repo DIR] [--ref BRANCH] [--dry-run]
  sync.py record --harness NAME [--repo DIR] (SKILL | SKILL/FILE ... | --all)
  sync.py status --harness NAME [--repo DIR] [--since REV]

sync    Adds new upstream skills, updates changed ones, removes retired or excluded
        ones. Each file renders as: upstream -> rename rewrites -> global transforms
        -> file transforms (transform.json). A skill whose upstream tree and
        transforms hash are unchanged is skipped without reading any blobs.
        Prints a JSON report for the agent.
record  Stores the installed content of the given skills or files as file transforms
        (find/replace hunks against the rendered base), accepts the markers left in
        them, and updates manifest hashes. Run after adapting files by hand.
status  Prints a human-readable summary of what changed since REV (default HEAD) in
        the skills repository: transforms first, then new, updated, and removed
        skills, then outstanding adaptation and drift.

State (in pstack-sync/):
  needs.json            features pstack relies on; markers per need (harness-independent).
  harness/<harness>.md  harness profile: destination, and one row per need with support
                        native | keep | substitute | partial | none. Markers of needs
                        marked native or keep are accepted automatically.
  manifest.json         upstream commit, harness, exclude list; per skill: dir, upstream
                        tree, transforms hash, per file upstream blob, installed blob,
                        pending adaptation.
  transform.json        per harness: "accept" marker regexes, "global" regex transforms,
                        "files" keyed by upstream path "<skill>/<rel>".
"""
import argparse, difflib, fnmatch, hashlib, json, os, re, shutil, subprocess, sys

UPSTREAM = os.environ.get("PSTACK_SYNC_UPSTREAM", "https://github.com/cursor/plugins")
SUBDIR = "pstack/skills"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.dirname(os.path.dirname(HERE))
SELF = "pstack-sync"
CACHE = os.path.expanduser("~/.cache/pstack-sync/%s.git" % hashlib.sha1(UPSTREAM.encode()).hexdigest()[:12])
SUPPORT = ("native", "keep", "substitute", "partial", "none")


# ---------- git / upstream ----------

def git(*args, input=None):
    r = subprocess.run(["git", "--git-dir", CACHE, *args], input=input, capture_output=True, check=True)
    return r.stdout


def fetch(ref):
    if not os.path.isdir(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        subprocess.run(["git", "init", "-q", "--bare", CACHE], check=True)
        git("remote", "add", "origin", UPSTREAM)
    git("fetch", "-q", "--depth", "1", "origin", ref)
    return git("rev-parse", "FETCH_HEAD").decode().strip()


def ensure_commit(sha):
    if subprocess.run(["git", "--git-dir", CACHE, "cat-file", "-e", f"{sha}^{{commit}}"],
                      capture_output=True).returncode:
        git("fetch", "-q", "--depth", "1", "origin", sha)


def upstream_skills(sha):
    """{name: {"tree": sha, "files": {rel: (mode, blob)}}} from one ls-tree call."""
    skills = {}
    for line in git("ls-tree", "-r", "-t", sha, f"{SUBDIR}/").decode().splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, obj = meta.split()
        parts = path[len(SUBDIR) + 1:].split("/", 1)
        if kind == "tree" and len(parts) == 1:
            skills.setdefault(parts[0], {"files": {}})["tree"] = obj
        elif kind == "blob" and len(parts) == 2:
            skills.setdefault(parts[0], {"files": {}})["files"][parts[1]] = (mode, obj)
    return {n: s for n, s in skills.items() if "SKILL.md" in s["files"]}


def read_blobs(shas):
    shas = list(dict.fromkeys(shas))
    if not shas:
        return {}
    out, data, pos = {}, git("cat-file", "--batch", input=("\n".join(shas) + "\n").encode()), 0
    for sha in shas:
        nl = data.index(b"\n", pos)
        size = int(data[pos:nl].split()[2])
        out[sha] = data[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
    return out


def blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------- needs and harness profile ----------

def load_needs(repo):
    return json.load(open(os.path.join(repo, SELF, "needs.json")))["needs"]


def parse_profile(text):
    """{need id: support} from rows like | `subagents` | substitute | ... |"""
    rows = {}
    for m in re.finditer(r"(?m)^\|\s*`?([\w-]+)`?\s*\|\s*(%s)\b" % "|".join(SUPPORT), text or ""):
        rows[m.group(1)] = m.group(2)
    return rows


def needs_in(text, needs):
    """{need id: sorted markers} found in text."""
    found = {}
    for n in needs:
        hits = {m.group(0) for p in n["markers"] for m in re.finditer(p, text)}
        if hits:
            found[n["id"]] = sorted(hits)
    return found


def flatten(found):
    return sorted({m for ms in found.values() for m in ms})


# ---------- rendering ----------

def rewrite_refs(text, renames):
    for old, new in renames.items():
        o = re.escape(old)
        text = re.sub(rf"(?<![\w./-])/{o}(?![\w-])", f"/{new}", text)
        text = re.sub(rf"\*\*{o}\*\*", f"**{new}**", text)
        text = re.sub(rf"`{o}`", f"`{new}`", text)
        text = re.sub(rf"\.\./{o}/", f"../{new}/", text)
    return text


def set_name(text, name):
    if not text.startswith("---"):
        return f"---\nname: {name}\n---\n\n{text}"
    end = text.index("\n---", 3)
    fm, body = text[:end], text[end:]
    if re.search(r"(?m)^name:.*$", fm):
        fm = re.sub(r"(?m)^name:.*$", f"name: {name}", fm, count=1)
    else:
        fm = fm.replace("---", f"---\nname: {name}", 1)
    return fm + body


def base_render(key, rel, data, target, renames, tf):
    """Upstream text through rename rewrites and global transforms; None if binary."""
    if b"\0" in data[:8192]:
        return None
    text = data.decode("utf-8")
    if rel.endswith(".md"):
        text = rewrite_refs(text, renames)
    if rel == "SKILL.md":
        text = set_name(text, target)
    for g in tf.get("global", []):
        if fnmatch.fnmatch(key, g.get("files", "*")):
            text = re.sub(g["pattern"], g["replace"], text)
    return text


def apply_file_transforms(text, transforms):
    """Returns (text or None if deleted, list of stale transforms)."""
    stale = []
    for t in transforms:
        if t.get("delete"):
            return None, []
        if text.count(t["find"]) == 1:
            text = text.replace(t["find"], t["replace"])
        else:
            stale.append({"find": t["find"], "replace": t["replace"]})
    return text, stale


def upstream_diff(old, new, path):
    """Unified diff of a file's rendered upstream text between two syncs."""
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        f"a/{path}", f"b/{path}", n=1))


def hunks(base, new):
    """Minimal unique find/replace hunks turning base into new."""
    a, b = base.splitlines(keepends=True), new.splitlines(keepends=True)
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    ranges = []
    for _, i1, i2, j1, j2 in ops:
        lo, hi = i1, i2
        while True:
            find = "".join(a[lo:hi])
            if find and base.count(find) == 1:
                break
            if lo == 0 and hi == len(a):
                break
            lo, hi = max(lo - 1, 0), min(hi + 1, len(a))
        if ranges and lo <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], hi)
            ranges[-1][2].append((i1, i2, j1, j2))
        else:
            ranges.append([lo, hi, [(i1, i2, j1, j2)]])
    out = []
    for lo, hi, group in ranges:
        rep, cur = [], lo
        for i1, i2, j1, j2 in group:
            rep += a[cur:i1] + b[j1:j2]
            cur = i2
        rep += a[cur:hi]
        out.append({"find": "".join(a[lo:hi]), "replace": "".join(rep)})
    return out


# ---------- state ----------

def load_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def norm_skills(skills):
    return {k: (v if isinstance(v, dict) else {"dir": v}) for k, v in skills.items()}


class State:
    def __init__(self, repo, harness):
        self.repo = repo
        self.harness = harness
        self.manifest_path = os.path.join(repo, SELF, "manifest.json")
        self.transform_path = os.path.join(repo, SELF, "transform.json")
        self.profile_path = os.path.join(repo, SELF, "harness", f"{harness}.md")
        self.manifest = load_json(self.manifest_path, {"skills": {}})
        self.transforms = load_json(self.transform_path, {})
        self.tf = self.transforms.setdefault(harness, {"accept": [], "global": [], "files": {}})
        self.tf.setdefault("files", {})
        self.managed = norm_skills(self.manifest["skills"])
        self.exclude = set(self.manifest.get("exclude", []))
        self.needs = load_needs(repo)
        self.profile_exists = os.path.exists(self.profile_path)
        self.profile = parse_profile(open(self.profile_path).read() if self.profile_exists else "")
        self.auto_accept = {i for i, s in self.profile.items() if s in ("native", "keep")}

    def skill_tf_hash(self, name, renames):
        files = {k: v for k, v in self.tf.get("files", {}).items() if k.startswith(name + "/")}
        blob = json.dumps([self.tf.get("global", []), self.tf.get("accept", []), files, renames,
                           sorted(self.auto_accept), self.needs], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()

    def outstanding(self, text, accepted):
        """Needs with markers not accepted by profile, accept regexes, or the file's list."""
        out = {}
        for nid, ms in needs_in(text, self.needs).items():
            if nid in self.auto_accept:
                continue
            ms = [m for m in ms if m not in accepted and not any(re.search(p, m) for p in self.tf.get("accept", []))]
            if ms:
                out[nid] = ms
        return out

    def targets(self, upstream):
        managed_dirs = {v["dir"] for v in self.managed.values()}
        existing = {d for d in os.listdir(self.repo) if os.path.isfile(os.path.join(self.repo, d, "SKILL.md"))}
        foreign = existing - managed_dirs - {SELF}
        targets = {}
        for name in upstream:
            if name in self.managed:
                targets[name] = self.managed[name]["dir"]
            elif name in foreign or name == SELF:
                targets[name] = f"pstack-{name}"
            else:
                targets[name] = name
            if targets[name] in foreign:
                sys.exit(f"error: {name} -> {targets[name]} collides with a non-pstack skill")
        return targets


def read_installed(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


# ---------- commands ----------

def cmd_sync(a, st):
    sha = fetch(a.ref)
    ups = {n: s for n, s in upstream_skills(sha).items() if n not in st.exclude}
    targets = st.targets(ups)
    renames = {u: t for u, t in targets.items() if u != t}
    harness_changed = st.manifest.get("harness") != a.harness

    report = {"upstream_commit": sha, "previous_commit": st.manifest.get("commit"),
              "harness": a.harness, "harness_changed": harness_changed,
              "profile_missing": not st.profile_exists, "profile_gaps": [], "dry_run": a.dry_run,
              "added": [], "updated": [], "removed": [], "unchanged_count": 0,
              "renamed_for_collision": renames, "skipped_binary_files": [], "local_drift": [],
              "needs_security_review": [], "needs_adaptation": {}}
    new_skills = {}
    prev_fetched = []

    def previous_blob(obj):
        """A blob from the previous sync's commit, or None if it can no longer be fetched."""
        if not prev_fetched and st.manifest.get("commit"):
            prev_fetched.append(True)
            try:
                ensure_commit(st.manifest["commit"])
            except subprocess.CalledProcessError:
                pass
        try:
            return read_blobs([obj])[obj]
        except (subprocess.CalledProcessError, ValueError, IndexError):
            return None

    def flag(target, rel, pending):
        report["needs_adaptation"].setdefault(target, {})[rel] = pending

    for name, up in sorted(ups.items()):
        target, entry = targets[name], st.managed.get(name, {})
        dst = os.path.join(st.repo, target)
        tfh = st.skill_tf_hash(name, renames)
        if (not harness_changed and entry.get("tree") == up["tree"]
                and entry.get("transforms") == tfh and os.path.isdir(dst)):
            report["unchanged_count"] += 1
            new_skills[name] = entry
            for rel, f in entry.get("files", {}).items():
                if f.get("pending"):
                    flag(target, rel, f["pending"])
            continue

        blobs = read_blobs(obj for _, obj in up["files"].values())
        files, changed = {}, not os.path.isdir(dst)
        recorded = entry.get("files", {})
        for rel, (mode, obj) in sorted(up["files"].items()):
            key = f"{name}/{rel}"
            base = base_render(key, rel, blobs[obj], target, renames, st.tf)
            if base is None:
                report["skipped_binary_files"].append(f"{target}/{rel}")
                continue
            ftf = st.tf["files"].get(key, {})
            text, stale = apply_file_transforms(base, ftf.get("transforms", []))
            path = os.path.join(dst, rel)
            current = read_installed(path)
            if (current is not None and rel in recorded
                    and blob_sha(current) != recorded[rel].get("installed")):
                report["local_drift"].append(f"{target}/{rel}")
                files[rel] = {**recorded[rel], "upstream": obj}
                continue
            if text is None:
                if current is not None:
                    changed = True
                    if not a.dry_run:
                        os.remove(path)
                continue
            new = text.encode("utf-8")
            todo = st.outstanding(text, ftf.get("accepted", []))
            pending = {k: v for k, v in (("stale_transforms", stale), ("needs", todo)) if v}
            if pending:
                flag(target, rel, pending)
                old_obj = recorded.get(rel, {}).get("upstream")
                if stale and old_obj and old_obj != obj:
                    old = previous_blob(old_obj)
                    if old is not None and b"\0" not in old[:8192]:
                        old_base = base_render(key, rel, old, target, renames, st.tf)
                        report["needs_adaptation"][target][rel] = {
                            **pending, "upstream_diff": upstream_diff(old_base, base, f"{target}/{rel}")}
            if current != new:
                changed = True
                if rel.startswith("scripts/") or rel == "mcp.json" or (
                        rel == "SKILL.md" and re.search(r"(?m)^mcpServers:", text)):
                    report["needs_security_review"].append(f"{target}/{rel}")
                if not a.dry_run:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(new)
                    os.chmod(path, 0o755 if mode == "100755" else 0o644)
            files[rel] = {"upstream": obj, "installed": blob_sha(new)}
            if pending:
                files[rel]["pending"] = pending

        if os.path.isdir(dst):  # files present locally but gone upstream
            for dp, _, fns in os.walk(dst):
                for fn in fns:
                    rel = os.path.relpath(os.path.join(dp, fn), dst)
                    if rel not in up["files"]:
                        changed = True
                        if not a.dry_run:
                            os.remove(os.path.join(dp, fn))

        if not entry:
            report["added"].append(target)
        elif changed:
            report["updated"].append(target)
        else:
            report["unchanged_count"] += 1
        new_skills[name] = {"dir": target, "tree": up["tree"], "transforms": tfh, "files": files}

    for name, entry in sorted(st.managed.items()):
        if name not in targets:
            report["removed"].append(entry["dir"])
            if not a.dry_run:
                shutil.rmtree(os.path.join(st.repo, entry["dir"]), ignore_errors=True)

    flagged = {nid for files in report["needs_adaptation"].values() for p in files.values()
               for nid in p.get("needs", {})}
    report["profile_gaps"] = sorted(flagged - set(st.profile))

    if not a.dry_run:
        lic = read_blobs([git("rev-parse", f"{sha}:pstack/LICENSE").decode().strip()])
        with open(os.path.join(st.repo, SELF, "LICENSE-pstack"), "wb") as f:
            f.write(next(iter(lic.values())))
        st.manifest = {"upstream": f"{UPSTREAM}/tree/{a.ref}/{SUBDIR}", "commit": sha,
                       "harness": a.harness, "exclude": sorted(st.exclude), "skills": new_skills}
        save_json(st.manifest_path, st.manifest)
        save_json(st.transform_path, st.transforms)
    print(json.dumps(report, indent=2))


def cmd_record(a, st):
    sha = st.manifest.get("commit")
    if not sha or st.manifest.get("harness") != a.harness:
        sys.exit("error: run `sync --harness %s` first" % a.harness)
    ensure_commit(sha)
    ups = upstream_skills(sha)
    by_dir = {v["dir"]: k for k, v in st.managed.items()}
    picks = {}  # upstream name -> set of rels, or None for every file
    for arg in (sorted(st.managed) if a.all else a.skills):
        skill, _, rel = arg.partition("/")
        name = by_dir.get(skill, skill)
        if rel:
            if picks.get(name, set()) is not None:
                picks.setdefault(name, set()).add(rel)
        else:
            picks[name] = None
    renames = {u: v["dir"] for u, v in st.managed.items() if u != v["dir"]}
    summary = {}
    for name in sorted(picks):
        if name not in st.managed or name not in ups:
            sys.exit(f"error: {name} is not a managed upstream skill")
        entry, up = st.managed[name], ups[name]
        target = entry["dir"]
        dst = os.path.join(st.repo, target)
        blobs = read_blobs(obj for _, obj in up["files"].values())
        extra = [os.path.relpath(os.path.join(dp, fn), dst) for dp, _, fns in os.walk(dst) for fn in fns]
        extra = [r for r in extra if r not in up["files"]]
        if extra:
            sys.exit(f"error: {target} has files not in upstream: {extra}")
        files, counts = dict(entry.get("files", {})), {}
        for rel, (mode, obj) in sorted(up["files"].items()):
            if picks[name] is not None and rel not in picks[name]:
                continue
            key = f"{name}/{rel}"
            base = base_render(key, rel, blobs[obj], target, renames, st.tf)
            if base is None:
                continue
            current = read_installed(os.path.join(dst, rel))
            if current is None:
                st.tf["files"][key] = {"transforms": [{"delete": True}], "accepted": []}
                files.pop(rel, None)
                counts[rel] = "deleted"
                continue
            text = current.decode("utf-8")
            ts = hunks(base, text) if text != base else []
            rebuilt, stale = apply_file_transforms(base, ts)
            if stale or rebuilt != text:
                ts = [{"find": base, "replace": text}]  # whole-file fallback
            accepted = flatten(st.outstanding(text, []))
            if ts or accepted:
                st.tf["files"][key] = {"transforms": ts, "accepted": accepted}
                counts[rel] = {"transforms": len(ts), "accepted": accepted}
            else:
                st.tf["files"].pop(key, None)
            files[rel] = {"upstream": obj, "installed": blob_sha(current)}
        entry.update({"files": files, "transforms": st.skill_tf_hash(name, renames)})
        summary[target] = counts
    st.manifest["skills"] = st.managed
    save_json(st.manifest_path, st.manifest)
    save_json(st.transform_path, st.transforms)
    print(json.dumps({"recorded": summary}, indent=2))


def cmd_status(a, st):
    def at_rev(rel, default):
        r = subprocess.run(["git", "-C", st.repo, "show", f"{a.since}:{SELF}/{rel}"], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else default

    old_m = json.loads(at_rev("manifest.json", '{"skills": {}}'))
    old_tf = json.loads(at_rev("transform.json", "{}")).get(a.harness, {}).get("files", {})
    old_profile = parse_profile(at_rev(f"harness/{a.harness}.md", ""))
    old_s, new_s = norm_skills(old_m.get("skills", {})), st.managed
    new_tf = st.tf.get("files", {})
    dir_of = {**{k: v["dir"] for k, v in old_s.items()}, **{k: v["dir"] for k, v in new_s.items()}}

    def upstream_changed(key):
        name, rel = key.split("/", 1)
        o = old_s.get(name, {}).get("files", {}).get(rel, {}).get("upstream")
        n = new_s.get(name, {}).get("files", {}).get(rel, {}).get("upstream")
        return o != n

    def addressed(ts):
        """Needs whose markers an edit removes."""
        ids = set()
        for t in ts:
            if t.get("delete"):
                continue
            before, after = needs_in(t["find"], st.needs), needs_in(t["replace"], st.needs)
            ids |= {i for i, ms in before.items() if set(ms) - set(after.get(i, []))}
        return sorted(ids)

    def describe(key, ts):
        name, rel = key.split("/", 1)
        if any(t.get("delete") for t in ts):
            what = "file removed"
        else:
            what = "%d edit%s" % (len(ts), "" if len(ts) == 1 else "s")
        ids = addressed(ts)
        return f"{dir_of.get(name, name)}/{rel}: {what}" + (f" (needs: {', '.join(ids)})" if ids else "")

    groups = {"new": [], "changed": [], "replayed": [], "dropped": []}
    kept = 0
    for key in sorted(set(old_tf) | set(new_tf)):
        o, n = old_tf.get(key, {}).get("transforms", []), new_tf.get(key, {}).get("transforms", [])
        if key.split("/", 1)[0] not in new_s:
            continue
        if n and not o:
            groups["new"].append(describe(key, n))
        elif o and not n:
            groups["dropped"].append(describe(key, o))
        elif n and o != n:
            groups["changed"].append(describe(key, n))
        elif n and upstream_changed(key):
            groups["replayed"].append(describe(key, n))
        elif n:
            kept += 1

    added = sorted(new_s[k]["dir"] for k in set(new_s) - set(old_s))
    removed = sorted(old_s[k]["dir"] for k in set(old_s) - set(new_s))
    updated = []
    for k in sorted(set(new_s) & set(old_s)):
        if new_s[k].get("tree") != old_s[k].get("tree"):
            n = sum(1 for rel in new_s[k].get("files", {}) if upstream_changed(f"{k}/{rel}"))
            updated.append(f"{new_s[k]['dir']} ({n} file{'s' if n != 1 else ''})")

    pending, drift = [], []
    for k, e in sorted(new_s.items()):
        for rel, f in sorted(e.get("files", {}).items()):
            if f.get("pending"):
                p = f["pending"]
                bits = list(p.get("needs", {}))
                if p.get("stale_transforms"):
                    bits.append("stale transform")
                pending.append(f"{e['dir']}/{rel}: {', '.join(bits)}")
            cur = read_installed(os.path.join(st.repo, e["dir"], rel))
            if cur is not None and blob_sha(cur) != f.get("installed"):
                drift.append(f"{e['dir']}/{rel}")

    prof_new = sorted(i for i in st.profile if i not in old_profile)
    prof_changed = sorted(i for i in st.profile if i in old_profile and old_profile[i] != st.profile[i])

    out = [f"# pstack sync status ({a.harness})", ""]
    oc, nc = (old_m.get("commit") or "none")[:7], (st.manifest.get("commit") or "none")[:7]
    out.append(f"Upstream cursor/plugins: {oc} -> {nc}" if oc != nc else f"Upstream cursor/plugins: {nc} (unchanged)")
    out.append("")
    total = sum(len(v) for v in groups.values())
    counts = ", ".join(f"{len(groups[g])} {g}" for g in ("new", "changed", "replayed", "dropped") if groups[g])
    out.append(f"## Transforms: {total} file{'s' if total != 1 else ''} touched" + (f" ({counts})" if counts else "")
               + f", {kept} carried over unchanged")
    labels = {"new": "New adaptations", "changed": "Changed adaptations",
              "replayed": "Replayed onto new upstream text", "dropped": "Dropped adaptations"}
    for g in ("new", "changed", "replayed", "dropped"):
        if groups[g]:
            out.append(f"{labels[g]} ({len(groups[g])}):")
            out += [f"- {x}" for x in groups[g]]
    if not total:
        out.append("No transform changes.")
    out.append("")
    out.append("## Skills")
    for label, items in (("New", added), ("Updated upstream", updated), ("Removed", removed)):
        out.append(f"{label} ({len(items)}): " + (", ".join(items) if items else "none"))
    if prof_new or prof_changed:
        out.append("")
        out.append("## Harness profile")
        if prof_new:
            out.append("Needs mapped: " + ", ".join(f"{i} ({st.profile[i]})" for i in prof_new))
        if prof_changed:
            out.append("Support changed: " + ", ".join(f"{i} {old_profile[i]} -> {st.profile[i]}" for i in prof_changed))
    out.append("")
    out.append(f"## Outstanding ({len(pending)} file{'s need' if len(pending) != 1 else ' needs'} adaptation, {len(drift)} drifted)")
    out += [f"- {x}" for x in pending] or ["Nothing outstanding."]
    out += [f"- drift: {x}" for x in drift]
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("sync", "record", "status"):
        p = sub.add_parser(c)
        p.add_argument("--harness", required=True, help="agent harness the skills are adapted for, e.g. amp")
        p.add_argument("--repo", default=DEFAULT_REPO)
    sub.choices["sync"].add_argument("--ref", default="main")
    sub.choices["sync"].add_argument("--dry-run", action="store_true")
    sub.choices["record"].add_argument("skills", nargs="*")
    sub.choices["record"].add_argument("--all", action="store_true")
    sub.choices["status"].add_argument("--since", default="HEAD")
    a = ap.parse_args()
    st = State(os.path.abspath(a.repo), a.harness)
    {"sync": cmd_sync, "record": cmd_record, "status": cmd_status}[a.cmd](a, st)


if __name__ == "__main__":
    main()
