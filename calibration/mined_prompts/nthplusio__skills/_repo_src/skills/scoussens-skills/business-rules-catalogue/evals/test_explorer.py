"""Opt-in Chromium contracts for the generated site and single-file explorer.

BR_BROWSER_TEST=1 python3 -B -m unittest discover -s <skill>/evals -p 'test_*.py' -v
Requires agent-browser and its installed Chromium. No host or network is used.
"""

import json
import os
import shutil
import subprocess
import time
import unittest

import test_br


@unittest.skipUnless(os.environ.get("BR_BROWSER_TEST") == "1" and shutil.which("agent-browser"),
                     "set BR_BROWSER_TEST=1 with agent-browser installed")
class ExplorerBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_br.BrEndToEnd.setUpClass()
        cls.addClassCleanup(test_br.BrEndToEnd.tearDownClass)
        test_br.BrEndToEnd("test_full_run").test_full_run()
        cls.work = test_br.BrEndToEnd.work
        cls.downloads = cls.work / "downloads"
        cls.downloads.mkdir()
        cls.session = f"br-eval-{os.getpid()}"
        cls.addClassCleanup(subprocess.run, ["agent-browser", "--session", cls.session, "close"],
                            capture_output=True, timeout=60)

    def browser(self, *args):
        p = subprocess.run(["agent-browser", "--session", self.session,
                            "--download-path", str(self.downloads), *args, "--json"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr or p.stdout)
        result = json.loads(p.stdout)
        self.assertTrue(result["success"], result)
        return result["data"]

    def evaluate(self, code):
        return self.browser("eval", code)["result"]

    def test_review_exports_navigation_and_imports(self):
        for entry in (self.work / "site/index.html", self.work / "bundle.html"):
            with self.subTest(entry=entry.name):
                self.browser("open", entry.as_uri())
                self.evaluate("localStorage.clear()")
                self.browser("reload")
                self.evaluate('location.hash = "rule=" + BR_DATA.rules[0].id')
                self.browser("wait", "#note")
                result = self.evaluate("""(async () => {
                  const ids = BR_DATA.rules.map(r => r.id);
                  const key = `br:${BR_DATA.meta.project.slug}:reviews:${BR_DATA.meta.commit.slice(0,9)}`;
                  const read = () => JSON.parse(localStorage.getItem(key));
                  const input = document.querySelector('#reviewer');
                  input.value = 'Product architect'; input.dispatchEvent(new Event('change'));
                  document.querySelector('[data-verdict=wrong]').click();
                  const oldCreate = URL.createObjectURL, oldClick = HTMLAnchorElement.prototype.click;
                  const blobs = [];
                  URL.createObjectURL = b => { blobs.push(b); return 'blob:test'; };
                  HTMLAnchorElement.prototype.click = function() {};
                  const exports = {};
                  const type = text => {
                    const note = document.querySelector('#note');
                    note.value = text; note.dispatchEvent(new Event('input', {bubbles:true}));
                  };
                  for (const format of ['json','csv','md']) {
                    type(`Latest ${format} correction, "quoted"\nSecond line`);
                    document.querySelector(`[data-export=${format}]`).click();
                    exports[format] = await blobs.at(-1).text();
                  }
                  URL.createObjectURL = oldCreate; HTMLAnchorElement.prototype.click = oldClick;
                  type('This note belongs to the first rule');
                  document.querySelector('[data-nav=next]').click();
                  await new Promise(r => setTimeout(r, 600));
                  return {exports, reviews:read(), first:ids[0], second:ids[1],
                    selected:new URLSearchParams(location.hash.slice(1)).get('rule')};
                })()""")
                exported = json.loads(result["exports"]["json"])
                first = result["first"]
                self.assertEqual(exported["reviews"][first]["note"], 'Latest json correction, "quoted"\nSecond line')
                self.assertEqual(exported["reviews"][first]["verdict"], "wrong")
                self.assertEqual(exported["reviews"][first]["reviewer"], "Product architect")
                self.assertIn('"Latest csv correction, ""quoted""\nSecond line"', result["exports"]["csv"])
                self.assertIn('Latest md correction, "quoted"\nSecond line', result["exports"]["md"])
                self.assertIn('Module: `services/orders/src/shipping.ts`', result["exports"]["md"])
                self.assertEqual(result["reviews"][first]["note"], "This note belongs to the first rule")
                self.assertNotIn(result["second"], result["reviews"])
                self.assertEqual(result["selected"], result["second"])
                self.browser("reload")
                self.evaluate(f'location.hash = "rule={first}"')
                self.browser("wait", "#note")
                self.assertEqual(self.evaluate("document.querySelector('#note').value"), "This note belongs to the first rule")

                merge = self.evaluate("""(async () => {
                  const id = BR_DATA.rules[0].id;
                  const key = `br:${BR_DATA.meta.project.slug}:reviews:${BR_DATA.meta.commit.slice(0,9)}`;
                  const read = () => JSON.parse(localStorage.getItem(key));
                  const original = read()[id];
                  const make = (at, note) => ({commit:BR_DATA.meta.commit, project:BR_DATA.meta.project.slug,
                    reviews:{[id]:{verdict:'unsure', note, reviewer:'Business owner', at}}});
                  const load = async body => {
                    const transfer = new DataTransfer();
                    transfer.items.add(new File([JSON.stringify(body)], 'review.json', {type:'application/json'}));
                    const file = document.querySelector('#importFile');
                    file.files = transfer.files; file.dispatchEvent(new Event('change'));
                    await new Promise(r => setTimeout(r, 100));
                    return read()[id];
                  };
                  const older = await load(make('2000-01-01T00:00:00Z','Old correction'));
                  const newer = await load(make('2099-01-01T00:00:00Z','New business correction'));
                  const otherCommit = await load({...make('2100-01-01T00:00:00Z','Bad commit'), commit:'other'});
                  const missingCommit = make('2100-01-01T00:00:00Z','Missing commit'); delete missingCommit.commit;
                  const missing = await load(missingCommit);
                  const otherProject = await load({...make('2100-01-01T00:00:00Z','Bad project'),project:'other'});
                  const malformed = make('2100-01-01T00:00:00Z','Bad verdict'); malformed.reviews[id].verdict='bogus';
                  const bad = await load(malformed);
                  const unknown = make('2100-01-01T00:00:00Z','Unknown rule'); unknown.reviews.unknown=unknown.reviews[id];
                  const mixed = await load(unknown);
                  return {original,older,newer,otherCommit,missing,otherProject,bad,mixed};
                })()""")
                self.assertEqual(merge["older"], merge["original"])
                self.assertEqual(merge["newer"]["note"], "New business correction")
                for field in ("otherCommit", "missing", "otherProject", "bad", "mixed"):
                    self.assertEqual(merge[field], merge["newer"])
                self.assertEqual(self.browser("errors")["errors"], [])

    def test_keyboard_tree_download_import_and_narrow_layout(self):
        self.browser("open", (self.work / "site/index.html").as_uri())
        self.evaluate("localStorage.clear()")
        self.browser("reload")
        self.browser("set", "viewport", "1440", "900", "2")
        self.browser("focus", "#tree")
        self.browser("press", "ArrowDown")
        self.browser("press", "ArrowRight")
        self.browser("press", "ArrowDown")
        self.browser("press", "ArrowDown")
        self.browser("press", "ArrowRight")
        self.browser("press", "ArrowDown")
        self.browser("press", "Enter")
        self.browser("wait", "#note")
        self.assertEqual(self.evaluate("document.querySelector('#detail h2').textContent"), "Rule free-shipping")
        self.browser("fill", "#reviewer", "Business owner")
        self.browser("press", "Tab")
        self.browser("focus", "[data-verdict=wrong]")
        self.browser("press", "Enter")
        self.browser("fill", "#note", "Charge 4.95 below 99.99")
        self.browser("focus", "#exportToggle")
        self.browser("press", "Enter")
        self.browser("press", "Tab")
        self.assertEqual(self.evaluate("document.activeElement.dataset.export"), "csv")
        self.browser("press", "Enter")
        deadline = time.monotonic() + 5
        while not list(self.downloads.glob("*.csv")) and time.monotonic() < deadline:
            time.sleep(.05)
        paths = list(self.downloads.glob("*.csv"))
        self.assertEqual(len(paths), 1)
        self.assertIn("Charge 4.95 below 99.99", paths[0].read_text(encoding="utf-8-sig"))
        self.browser("focus", "#exportToggle")
        self.browser("press", "Enter")
        for _ in range(4):
            self.browser("press", "Tab")
        self.assertEqual(self.evaluate("document.activeElement.id"), "importFile")
        catalogue = json.loads((self.work / "build/catalogue.json").read_text())
        imported = self.work / "keyboard-review.json"
        imported.write_text(json.dumps({"commit": catalogue["meta"]["commit"], "project": "shop", "reviews": {
            "BR-OR-001": {"verdict": "unsure", "note": "Owner correction from JSON",
                          "reviewer": "Business owner", "at": "2099-01-01T00:00:00Z"}}}))
        self.browser("upload", "#importFile", str(imported))
        self.browser("wait", "--fn", "document.querySelector('#toast').textContent === 'Imported 1 review'")
        self.assertEqual(self.evaluate("document.querySelector('#note').value"), "Owner correction from JSON")
        self.browser("press", "Escape")
        for width in (1440, 390):
            self.browser("set", "viewport", str(width), "900", "2")
            sizes = self.evaluate("""({width:innerWidth,scroll:document.documentElement.scrollWidth,
              loaded:BR_DATA.loaded.length,expected:BR_DATA.expected,
              noteBottom:document.querySelector('#note').getBoundingClientRect().bottom,
              reviewTop:document.querySelector('.review').getBoundingClientRect().top})""")
            self.assertLessEqual(sizes["scroll"], sizes["width"])
            self.assertEqual(sizes["loaded"], sizes["expected"])
            self.assertGreaterEqual(sizes["reviewTop"], sizes["noteBottom"])
        self.assertEqual(self.browser("errors")["errors"], [])


if __name__ == "__main__":
    unittest.main()
