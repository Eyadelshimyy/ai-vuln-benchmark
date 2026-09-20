# Juliet Test Suite calibration (Java)

The hand-built calibration set under `calibration/python_handbuilt/` covers
Python. For Java, this project uses NIST's **Juliet Test Suite for Java**
(part of the SARD -- Software Assurance Reference Dataset) as ground truth
instead of hand-writing pairs, because it is public domain, purpose-built
for exactly this (labeled `bad()`/`good()` method pairs per CWE test case),
and already cited by several of the papers in the literature review
(`Literature_Review_Sheet.tex`), which strengthens the methodological
defensibility of using it here.

**Not included in this repo:** the Juliet Test Suite is large (Java 1.3
alone is several hundred MB), so it isn't vendored here. Download it
yourself:

```
wget https://samate.nist.gov/SARD/downloads/test-suites/2023-01-25-juliet-test-suite-for-java-1-3.zip
unzip 2023-01-25-juliet-test-suite-for-java-1-3.zip -d juliet_java
```

(If that exact filename has moved, browse
https://samate.nist.gov/SARD/test-suites and search "Juliet Java".)

## Relevant CWE test case directories

Once unzipped, the test cases for this project's three CWEs live under:

```
juliet_java/src/testcases/CWE89_SQL_Injection/...
juliet_java/src/testcases/CWE78_OS_Command_Injection/...
juliet_java/src/testcases/CWE22_Path_Traversal/...
```

Each numbered test case (e.g. `CWE89_SQL_Injection__...`) contains a
`bad()` method (the vulnerable version) and one or more `good()` methods
(safe variants, sometimes split as `goodG2B()`/`goodB2G()` for
"good-source-bad-sink" / "bad-source-good-sink" data-flow directions).
That `bad`/`good` labeling is exactly the same ground-truth role that
`vulnerable_*.py` / `safe_*.py` plays in the hand-built Python set.

## How this plugs into the pipeline

1. Extract just the `bad()` and `good()` methods for the three target CWEs
   into standalone files (a short extraction script belongs here --
   `extract_juliet_cases.py`, not yet written -- since Juliet's own file
   structure bundles many CWE variants per file and needs to be split
   into single-vulnerability compilation units first, mirroring the one
   vulnerability per file contract the Python calibration set already
   uses).
2. Run `detection/run_detection.py` (a Java-language semgrep rule set
   analogous to `detection/semgrep_rules/python_*.yaml` needs to be added
   under `detection/semgrep_rules/java_*.yaml` first) against the
   extracted cases.
3. Run a Java exploit-confirmation harness analogous to
   `exploit_confirmation/harness_*.py` -- for Java this means compiling
   each extracted case with `javac` and driving it with JDBC/ProcessBuilder
   test doubles the same way the Python harnesses drive sqlite3/subprocess.
4. Compare against Juliet's own `bad`/`good` labels exactly the way
   `scripts/run_calibration.py` compares against the `vulnerable_*`/`safe_*`
   filename convention for Python.

This Java calibration step is scoped as the next concrete task after the
Python pipeline (proven working — see `scripts/run_calibration.py`,
12/12 = 100% accuracy against the hand-built ground truth) is extended
with real LLM-generated samples. It is deliberately kept as a documented
next step rather than partially implemented, since Java detection rules
and a JDBC/compile-based exploit harness are a meaningfully different
(and non-trivial) piece of engineering from the Python side and deserve
their own focused pass rather than a rushed placeholder.
