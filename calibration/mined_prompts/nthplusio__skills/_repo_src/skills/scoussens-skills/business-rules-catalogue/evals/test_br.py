"""Contract tests for scripts/br.py against a synthetic multi-language repository.

Run with `python3 -B -m unittest discover -s <skill>/evals -p 'test_*.py'`. No network; `node` is optional.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
BR = SKILL / "scripts" / "br.py"
spec = importlib.util.spec_from_file_location("br", BR)
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)

FILES = {
    "README.md": "# Shop\n",
    "docs/GLOSSARY.md": "# Glossary\n\n**Order**:\nA purchase a customer places.\n\n"
                        "**Invoice**:\nThe bill for one order.\n\n**Fee**:\nA charge added to an order.\n",
    "docs/adr/001-free-shipping.md": "# Free shipping above 100\n",
    "docs/executable-policy.yaml": "free_shipping_minimum: 99.99\n",
    "services/orders/package.json": '{"name": "orders"}\n',
    "services/orders/src/shipping.ts": "export function fee(total: number) {\n  if (total >= 99.99) return 0;\n"
                                       "  return 4.95;\n}\n",
    "services/orders/src/routes.ts": "import { fee } from './shipping';\nexport const routes = { post: fee };\n",
    "services/orders/src/shipping.test.ts": "test('fee', () => {});\n",
    "services/billing/go.mod": "module billing\n",
    "services/billing/invoice.go": "package billing\n\nfunc Due(days int) int {\n\tif days > 30 {\n"
                                   "\t\treturn 30\n\t}\n\treturn days\n}\n",
    "services/billing/invoice_test.go": "package billing\n",
    "db/migrations/001_init.sql": "CREATE TABLE orders (status TEXT DEFAULT 'NEW');\n",
    "infra/deploy.ts": "export const region = 'eu-west-1';\n",
    "web/package.json": '{"name": "web"}\n',
    "web/src/pages/checkout.tsx": "export default function Checkout() { return null; }\n",
}


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def rule(key, stage, statement, flags=(), example="", terms=()):
    return {"key": key, "title": f"Rule {key}", "stage": stage, "kind": "threshold", "statement": statement,
            "applies_when": "Always.", "exceptions": [], "example": example, "on_violation": None,
            "terms": list(terms), "surfaces": [{"kind": "api", "name": "POST /v1/orders",
                                                 "ref": "services/orders/src/routes.ts:2"}],
            "engineering": [{"role": "decides", "file": "services/orders/src/shipping.ts", "symbol": "fee",
                             "lines": "2-3"}],
            "flags": list(flags), "related": [], "confidence": "high"}


class BrEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="br-eval-"))
        cls.repo = cls.tmp / "shop"
        for path, text in FILES.items():
            p = cls.repo / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        git(cls.repo, "init", "-q")
        git(cls.repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "add", ".")
        git(cls.repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
        git(cls.repo, "remote", "add", "origin", "git@github.com:acme/shop.git")
        cls.work = cls.tmp / "work"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def br(self, *args, ok=True):
        res = subprocess.run([sys.executable, "-B", str(BR), *args], capture_output=True, text=True)
        if ok and res.returncode != 0:
            self.fail(f"br.py {' '.join(args)} failed:\n{res.stdout}\n{res.stderr}")
        return res

    def w(self, *args, ok=True):
        return self.br(args[0], "--work", str(self.work), *args[1:], ok=ok)

    def test_full_run(self):
        # init refuses a dirty tree and a work directory inside the repository.
        (self.repo / "infra/deploy.ts").write_text("changed\n")
        self.assertNotEqual(self.br("init", "--repo", str(self.repo), "--work", str(self.work), ok=False).returncode, 0)
        git(self.repo, "checkout", "--", "infra/deploy.ts")
        self.assertNotEqual(self.br("init", "--repo", str(self.repo), "--work", str(self.repo / "w"), ok=False).returncode, 0)
        self.br("init", "--repo", str(self.repo / "services"), "--work", str(self.work))
        run = json.loads((self.work / "run.json").read_text())
        self.assertEqual(run["links"]["file"], "https://github.com/acme/shop/blob/{commit}/{path}")

        # Changes after init cannot acquire citations to the old commit.
        (self.repo / "infra/deploy.ts").write_text("changed again\n")
        self.assertIn("Source changed", self.w("survey", ok=False).stderr)
        git(self.repo, "checkout", "--", "infra/deploy.ts")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "--allow-empty", "-qm", "moved")
        self.assertIn("Source changed", self.w("survey", ok=False).stderr)
        git(self.repo, "checkout", "--detach", run["commit"])

        # survey drafts areas from package roots and top-level folders, leaving tests out.
        self.w("survey")
        draft = json.loads((self.work / "scope.draft.json").read_text())
        areas = {a["paths"][0]: a for a in draft["areas"]}
        self.assertTrue({"services/orders", "services/billing", "db", "infra", "web"} <= set(areas))
        self.assertEqual(areas["services/orders"]["_survey"]["files"], 2)  # shipping.test.ts is left out
        self.assertEqual(areas["services/billing"]["_survey"]["files"], 1)
        self.assertEqual(draft["glossary"], {"path": "docs/GLOSSARY.md", "format": "bold-colon"})
        self.assertIn("docs/adr", draft["docs"])
        kinds = {k["id"] for k in draft["surfaceKinds"]}
        self.assertTrue({"api", "screen", "internal"} <= kinds)

        # The agent classifies the draft; survey then validates the scope.
        tiers = {"services/orders": "business", "services/billing": "business", "db": "business",
                 "infra": "platform", "web": "skip"}
        scope = dict(draft)
        scope["areas"] = [{**areas[p], "tier": t, "summary": f"{p} summary"} for p, t in tiers.items()]
        for a in scope["areas"]:
            a.pop("_survey")
        scope["stages"] = [
            {"id": "checkout", "label": "Checkout", "lane": "main", "description": "Placing an order.",
             "entries": ["POST /v1/orders"]},
            {"id": "billing", "label": "Billing", "lane": "main", "description": "Invoicing."},
            {"id": "platform", "label": "Platform", "lane": "band", "description": "Cross-cutting."},
        ]
        bad = json.loads(json.dumps(scope))
        bad["areas"][1]["code"] = bad["areas"][0]["code"]
        (self.work / "scope.json").write_text(json.dumps(bad))
        self.assertIn("is used twice", self.w("survey", ok=False).stdout)
        (self.work / "scope.json").write_text(json.dumps(scope))
        self.w("survey")
        codes = {a["id"]: a["code"] for a in scope["areas"]}

        # partition owns every in-scope file exactly once and renders prompts and the brief.
        nested_skip = {"id": "routing", "code": "RT", "title": "Routing", "tier": "skip",
                       "paths": ["services/orders/src/routes.ts"], "summary": "Routing only."}
        (self.work / "scope.json").write_text(json.dumps({**scope, "areas": scope["areas"] + [nested_skip]}))
        self.w("partition")
        self.assertNotIn("services/orders/src/routes.ts",
                         [l for f in (self.work / "files").glob("W*.txt") for l in f.read_text().splitlines()])
        (self.work / "scope.json").write_text(json.dumps(scope))
        self.w("partition", "--force")
        out = self.w("partition").stdout
        self.assertIn("0 errors", out)
        owned = sorted(l for f in (self.work / "files").glob("W*.txt") for l in f.read_text().split())
        self.assertEqual(owned, sorted(["services/orders/src/shipping.ts", "services/orders/src/routes.ts",
                                        "services/billing/invoice.go", "db/migrations/001_init.sql",
                                        "infra/deploy.ts"]))
        brief = (self.work / "BRIEF.md").read_text()
        self.assertIn("`docs/GLOSSARY.md` is the project's glossary", brief)
        self.assertIn("`checkout`: Placing an order.", brief)
        self.assertNotIn("{STAGES}", brief)
        self.assertIn(str(self.work / "BRIEF.md"), (self.work / "prompts" / "W1.md").read_text())

        # Reader outputs: the business reader and the platform summary reader.
        oid, bid, did, iid = (a["id"] for a in scope["areas"][:4])
        script = "Escapes `</script>` in a note."
        w1 = {"worker": "W1", "areas": [
            {"area": oid, "summary": None, "modules": [
                {"module": "services/orders/src/shipping.ts", "title": "Shipping fee", "stage": "checkout",
                 "summary": "Decides the fee.", "rules": [
                     rule("free-shipping", "checkout", "Orders of `99.99` or more ship free.", terms=["Order"],
                          flags=[{"type": "contradicts-docs", "detail": "ADR says 100.",
                                  "evidence": ["services/orders/src/shipping.ts:2", "docs/adr/001-free-shipping.md:1"]}]),
                     rule("flat-fee", "checkout", "Other orders pay `4.95`.", example=script)]},
                {"module": "services/orders/src/routes.ts", "title": "Routes", "stage": "checkout",
                 "summary": "Routes only.", "rules": []}]},
            {"area": bid, "modules": [{"module": "services/billing/invoice.go", "title": "Due date",
                                       "stage": "billing", "summary": "Caps terms.",
                                       "rules": [rule("terms-cap", "billing", "Terms cap at 30 days.", flags=[
                                           {"type": "inconsistent", "detail": "Checkout allows 45 days.",
                                            "evidence": ["services/billing/invoice.go:4", "services/orders/src/shipping.ts:2"]}])]}]},
            {"area": did, "modules": [{"module": "db/migrations/001_init.sql", "title": "Order defaults",
                                       "stage": "checkout", "summary": "Defaults.",
                                       "rules": [rule("new-status", "checkout", "New orders start `NEW`.")]}]}]}
        w2 = {"worker": "W2", "areas": [{"area": iid, "summary": "Deploys.", "modules": [
            {"module": "infra/deploy.ts", "title": "Region", "stage": "platform", "summary": "Region.",
             "rules": [rule("single-region", "platform", "Everything runs in `eu-west-1`.")]}]}]}
        (self.work / "out" / "W1.json").write_text(json.dumps(w1))
        (self.work / "out" / "W2.json").write_text(json.dumps(w2))
        self.assertIn("0 errors", self.w("check").stdout)

        for file, lines, role, message in (("missing.ts", "1", "decides", "tracked file"),
                                           ("services/orders/src/shipping.ts", "2-99", "decides", "outside the file"),
                                           ("services/orders/src/shipping.ts", "3-2", "decides", "outside the file"),
                                           ("services/orders/src/shipping.ts", "2-3", "calls", "no deciding code")):
            invalid = json.loads(json.dumps(w1))
            invalid["areas"][0]["modules"][0]["rules"][0]["engineering"] = [
                {"file": file, "lines": lines, "role": role}]
            (self.work / "out" / "W1.json").write_text(json.dumps(invalid))
            self.assertIn(message, self.w("check", ok=False).stdout)
        duplicate = json.loads(json.dumps(w1))
        duplicate["areas"][0]["modules"][1]["rules"] = [duplicate["areas"][0]["modules"][0]["rules"][0]]
        (self.work / "out" / "W1.json").write_text(json.dumps(duplicate))
        self.assertIn("area-unique key", self.w("check", ok=False).stdout)
        (self.work / "out" / "W1.json").write_text(json.dumps(w1))

        # A module a reader skipped shows up as a coverage warning; an unknown stage is an error.
        dropped = json.loads(json.dumps(w1))
        dropped["areas"][0]["modules"].pop()
        (self.work / "out" / "W1.json").write_text(json.dumps(dropped))
        self.assertIn("coverage: services/orders/src/routes.ts", self.w("check").stdout)
        dropped["areas"][0]["modules"][0]["rules"][0]["stage"] = "nowhere"
        (self.work / "out" / "W1.json").write_text(json.dumps(dropped))
        self.assertNotEqual(self.w("check", ok=False).returncode, 0)
        (self.work / "out" / "W1.json").write_text(json.dumps(w1))

        # Build requires a complete second read, not merely parseable output.
        self.assertIn("No second-read batches", self.w("build", ok=False).stderr)
        (self.work / "verify" / "out99.json").write_text("[]")
        self.w("batches")
        self.assertFalse((self.work / "verify" / "out99.json").exists())
        sampled = [x for f in (self.work / "verify").glob("in*.json") for x in json.loads(f.read_text())]
        self.assertIn("free-shipping", [x["key"] for x in sampled if not x.get("sample")])
        self.assertIn("Missing checker output", self.w("build", ok=False).stderr)
        for batch in sorted((self.work / "verify").glob("in*.json")):
            verdicts = []
            for item in json.loads(batch.read_text()):
                v = {k: item[k] for k in ("worker", "area", "key")}
                v.update(rule_verdict="correct", rule_revision=None,
                         flags=[{"index": i, "verdict": "confirmed"} for i in item["check_flags"]])
                if item["key"] == "free-shipping":
                    v.update(rule_verdict="needs-revision",
                             rule_revision={"statement": "Orders of `99.99` or more ship free, before tax."},
                             flags=[{"index": 0, "verdict": "refuted", "note": "The ADR was superseded."}])
                verdicts.append(v)
            batch.with_name(batch.name.replace("in", "out", 1)).write_text(json.dumps(verdicts))

        output = self.work / "verify" / "out1.json"
        good = json.loads(output.read_text())
        for bad in ([], good + good, [{**good[0], "flags": []}, *good[1:]],
                    [{**good[0], "rule_verdict": "unknown"}, *good[1:]]):
            output.write_text(json.dumps(bad))
            self.assertNotEqual(self.w("build", ok=False).returncode, 0)
        output.write_text(json.dumps(good))
        changed = json.loads(json.dumps(w1))
        changed["areas"][0]["modules"][0]["rules"][0]["statement"] = "Changed after sampling."
        (self.work / "out" / "W1.json").write_text(json.dumps(changed))
        self.assertIn("changed after sampling", self.w("build", ok=False).stderr)
        (self.work / "out" / "W1.json").write_text(json.dumps(w1))

        self.w("build")
        cat = json.loads((self.work / "build" / "catalogue.json").read_text())
        by_key = {r["key"]: r for r in cat["rules"]}
        self.assertEqual(len(cat["rules"]), 5)
        self.assertEqual(by_key["free-shipping"]["id"], f"BR-{codes[oid]}-001")
        self.assertEqual(by_key["free-shipping"]["statement"], "Orders of `99.99` or more ship free, before tax.")
        self.assertEqual(by_key["free-shipping"]["flags"], [])
        self.assertEqual(cat["meta"]["verification"]["flags_removed"], 1)
        self.assertEqual(cat["meta"]["verification"]["rules_checked"], 5)
        self.assertEqual(cat["glossary"], {"Order": "A purchase a customer places."})
        self.assertNotIn(iid, [a["id"] for a in cat["areas"] if a["tier"] == "skip"])
        self.assertEqual({a["id"] for a in cat["areas"]}, {oid, bid, did, iid})
        mod = next(m for m in cat["modules"] if m["path"] == "services/orders/src/shipping.ts")
        self.assertEqual((mod["folder"], mod["name"]), ("", "shipping.ts"))

        bundle = (self.work / "bundle.html").read_text()
        self.assertNotIn('<script src=', bundle)
        self.assertEqual(bundle.count("<script>"), bundle.lower().count("</script>"))
        if shutil.which("node"):
            names = [l.split('"')[1] for l in (self.work / "site" / "index.html").read_text().split("\n")
                     if l.startswith('<script src="data/')]
            loader = "global.window=global;" + "".join(f"require({json.dumps(str(self.work / 'site' / n))});"
                                                        for n in names)
            loader += "console.log(JSON.stringify([BR_DATA.loaded.length,BR_DATA.expected,BR_DATA.rules.length]))"
            loaded = json.loads(subprocess.check_output(["node", "-e", loader], text=True))
            self.assertEqual(loaded, [loaded[1], loaded[1], 5])
            subprocess.run(["node", "--check", str(self.work / "site" / "app.js")], check=True)

        # The note cites code at the commit and balances its code spans.
        note_path = self.tmp / "note.md"
        self.w("note", "--out", str(note_path))
        note = note_path.read_text()
        self.assertIn("What business rules does shop enforce?", note)
        self.assertIn("### 4.1 Contradicts docs\n\nNone.", note)  # the only such flag was refuted
        self.assertIn(f"[`services/billing/invoice.go:4`](https://github.com/acme/shop/blob/{run['commit']}/"
                      f"services/billing/invoice.go#L4-L4)", note.split("### 4.2")[1].split("## 5.")[0])

        # Hosted delivery needs a recorded destination first.
        self.assertNotEqual(self.w("publish-prep", ok=False).returncode, 0)
        self.w("record", "--destination", "docstash")
        self.assertNotEqual(self.w("publish-prep", ok=False).returncode, 0)
        self.w("record", "--destination", "docstash", "--audience", "private")
        out = self.w("publish-prep").stdout
        self.assertIn("audience: private", out)
        plan = json.loads((self.work / "publish" / "plan.json").read_text())
        self.assertEqual(sorted(i["file"] for g in plan for i in g["items"]),
                         sorted(f"data/{p.name}" for p in (self.work / "site" / "data").glob("*.js")))

        # A recorded share link reaches the note.
        self.w("record", "--share-url", "https://example.com/catalogue")
        self.w("note", "--out", str(note_path))
        self.assertIn("https://example.com/catalogue", note_path.read_text())
        self.assertIn("does not estimate the error rate among unchecked rules", note_path.read_text())

        # Explicit policy files can override both unknown format and skipped-folder heuristics.
        policy = {"id": "policy", "code": "PO", "title": "Policy", "tier": "business",
                  "paths": ["docs/executable-policy.yaml"], "summary": "Executable policy."}
        extra = {**scope, "areas": scope["areas"] + [policy], "include": ["docs/executable-policy.yaml"]}
        (self.work / "scope.json").write_text(json.dumps(extra))
        self.w("survey")
        self.w("partition", "--force")
        self.assertIn("docs/executable-policy.yaml",
                      [l for f in (self.work / "files").glob("W*.txt") for l in f.read_text().splitlines()])
        (self.work / "scope.json").write_text(json.dumps({**extra, "exclude": ["docs/executable-policy.yaml"]}))
        self.w("partition", "--force")
        self.assertNotIn("docs/executable-policy.yaml",
                         [l for f in (self.work / "files").glob("W*.txt") for l in f.read_text().splitlines()])
        (self.work / "scope.json").write_text(json.dumps(scope))
        self.w("partition", "--force")


class BrHelpers(unittest.TestCase):
    def test_codes_prefer_the_last_word(self):
        taken = set()
        self.assertEqual([br.unique_code(a, taken) for a in ("services-orders", "orders", "billing")],
                         ["OR", "ORD", "BI"])

    def test_grouped_module_strings_cover_their_files(self):
        self.assertTrue(br.covers("pkg/billing/_tax.py, _fees.py", "pkg/billing/_fees.py"))
        self.assertTrue(br.covers("pkg/ext/cloud (net, lb)", "pkg/ext/cloud/lb/main.py"))
        self.assertFalse(br.covers("pkg/ext/cloud (net, lb)", "pkg/ext/gcp/main.py"))

    def test_module_paths_relative_to_a_single_area_root_are_accepted(self):
        area = {"paths": ["packages/billing"]}
        self.assertEqual(br.module_path("billing/fees.py", area), "packages/billing/billing/fees.py")
        self.assertEqual(br.module_path("packages/billing/billing/fees.py", area), "packages/billing/billing/fees.py")

    def test_link_templates_follow_the_code_host(self):
        self.assertIsNone(br.code_url(None, "abc", "a.py:3"))
        gl = {"file": "https://gitlab.com/x/y/-/blob/{commit}/{path}", "line": "#L{start}-{end}", "commit": ""}
        self.assertEqual(br.code_url(gl, "abc", "a.py:3-9"), "https://gitlab.com/x/y/-/blob/abc/a.py#L3-9")
        self.assertEqual(br.code_url(gl, "abc", "a.py:3"), "https://gitlab.com/x/y/-/blob/abc/a.py#L3-3")


if __name__ == "__main__":
    unittest.main()
