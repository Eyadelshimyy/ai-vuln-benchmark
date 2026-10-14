"""Source-only contracts for the isolated CI routing requirement.

These contracts do not attest runner registration, physical isolation or actual
native/GPU execution. Those require the existing runner operator's receipts.
"""
from pathlib import Path
import re

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_JOBS = ("python-matrix", "python", "rust", "gpu-smoke", "fuzz", "package")
_CHECKOUT_JOBS = tuple(job for job in _JOBS if job != "python")


def _job(workflow: str, name: str) -> str:
    """Read one literal top-level job from the retained CI workflow grammar."""
    pattern = rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)"
    matches = re.findall(pattern, workflow, re.MULTILINE | re.DOTALL)
    assert len(matches) == 1, f"expected exactly one CI job: {name}"
    return matches[0]


@pytest.mark.parametrize("name", _JOBS)
def test_ci_jobs_require_isolated_linux_x64_self_hosted_routing(name: str) -> None:
    """Keep every current CI job off hosted/control/scanner fallback routes."""
    job = _job(_WORKFLOW.read_text(encoding="utf-8"), name)
    selectors = re.findall(r"^    runs-on: (.+)$", job, re.MULTILINE)
    assert selectors == ["[self-hosted, Linux, X64, cwlab-ci-isolated]"], (
        f"{name} must request the operator-provisioned isolated pool, without fallback"
    )


def _assert_checkout_credentials(job: str) -> None:
    """Check literal block steps; quotes/spacing cannot hide a second checkout.

    This is a bounded source contract, not a general YAML parser or a runtime
    credential attestation. Unsupported checkout syntax fails closed.
    """
    steps = re.findall(r"^      - [^\n]+\n.*?(?=^      - |\Z)", job, re.MULTILINE | re.DOTALL)
    checkouts = []
    for step in steps:
        uses = re.findall(
            r"^(?:      - |        )uses:[ \t]+(['\"]?)([^'\"\s]+)\1[ \t]*(?:#.*)?$",
            step,
            re.MULTILINE,
        )
        checkout_uses = [value for _, value in uses if value.startswith("actions/checkout@")]
        if "actions/checkout@" in step:
            assert len(checkout_uses) == 1 and len(uses) == 1, "unsupported checkout step syntax"
            checkouts.append(step)
    assert len(checkouts) == 1, "expected exactly one checkout per execution job"
    assert re.search(r"^        with:\n          persist-credentials: false$", checkouts[0], re.MULTILINE)


@pytest.mark.parametrize("name", _CHECKOUT_JOBS)
def test_ci_checkouts_do_not_persist_credentials(name: str) -> None:
    """Executable PR code must not inherit a checkout-persisted Git credential."""
    job = _job(_WORKFLOW.read_text(encoding="utf-8"), name)
    _assert_checkout_credentials(job)


def test_ci_keeps_all_existing_execution_job_identities() -> None:
    """Do not remove a failing required producer to satisfy the routing check."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert tuple(re.findall(r"^  ([A-Za-z0-9_-]+):$", workflow.split("jobs:\n", 1)[1], re.MULTILINE)) == _JOBS


def test_isolated_runner_label_is_declared_without_lint_exemptions() -> None:
    """Use normal actionlint discovery, never an external acceptance override.

    Basis: actionlint v1.7.12 configuration documentation, Configuration file
    and self-hosted-runner.labels sections (read after the initial RED).
    The exact minimal declaration retains every linter check. It establishes
    source configuration only, not eligible capacity or physical isolation.

    References:
        actionlint contributors. (n.d.). Configuration. actionlint
        (Version 1.7.12), Configuration file and self-hosted-runner.labels.
        https://github.com/rhysd/actionlint/blob/v1.7.12/docs/config.md
    """
    config = _ROOT / ".github" / "actionlint.yaml"
    assert config.is_file(), "declare the new isolated runner label for normal lint discovery"
    assert config.read_text(encoding="utf-8") == (
        "# Custom label syntax only; runner capacity and isolation need operator evidence.\n"
        "self-hosted-runner:\n"
        "  labels:\n"
        "    - cwlab-ci-isolated\n"
    ), "declare only the exact custom label; do not add ignores or wildcard exemptions"


# Checkout regression witnesses: source-only variants, no credentials or runner execution.
ACTION='actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'
VARIANTS=[('plain','', ' ',False),('two_spaces','','  ',False),('tab','','\t',False),
          ('single_quote',"'",' ',False),('double_quote','"',' ',False),('name_first','',' ',True)]
def step(quote,space,name_first,credential):
    action=quote+ACTION+quote
    prefix='      - name: Fixture checkout\n        uses:'+space+action+'\n' if name_first else '      - uses:'+space+action+'\n'
    return prefix+'        with:\n          persist-credentials: '+credential+'\n'
class Text:
    def __init__(self,text):self.text=text
    def read_text(self,**kwargs):return self.text
@pytest.mark.parametrize('name',_CHECKOUT_JOBS)
@pytest.mark.parametrize('variant,quote,space,name_first',VARIANTS)
@pytest.mark.parametrize('kind',['unsafe_second','safe_singleton'])
def test_checkout_guard_recognizes_literal_step_variants(monkeypatch,name,variant,quote,space,name_first,kind):
    base=_WORKFLOW.read_text(encoding='utf-8')
    job=_job(base,name)
    matches=list(re.finditer(r'^      - uses: actions/checkout@[^\n]+\n(.*?)(?=^      - |\Z)',job,re.M|re.S))
    assert len(matches)==1
    original=matches[0].group(0)
    assert 'persist-credentials: false' in original
    added=step(quote,space,name_first,'true' if kind=='unsafe_second' else 'false')
    replacement=original+added if kind=='unsafe_second' else added
    changed=job.replace(original,replacement,1)
    assert changed!=job or (kind=='safe_singleton' and variant=='plain')
    start=base.index(job);workflow=base[:start]+changed+base[start+len(job):]
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    # Execute original unchanged test function, not a lookalike parser.
    rejected=False
    try:test_ci_checkouts_do_not_persist_credentials(name)
    except AssertionError:rejected=True
    assert rejected is (kind=='unsafe_second'), (kind,variant,name,'original guard false admission/false rejection')
