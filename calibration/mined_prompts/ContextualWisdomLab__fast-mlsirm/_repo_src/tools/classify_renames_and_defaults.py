#!/usr/bin/env python3
"""Apply ADR-0028's naming/defaults policy to an inventory CSV.

Reads ``docs/api/inventory-<date>.csv`` (from ``tools/inventory_public_api.py``)
and writes ``docs/api/renames-and-defaults-<date>.csv``: one row per
(callable, defaulted-parameter) pair recording the ADR-0028 naming decision
for the callable and the default-argument decision for the parameter.
Callables with no defaulted parameters get a single row with the parameter
columns empty.

This is a mechanical, rule-based classification, not a per-function research
pass (ADR-0028, "Non-goals"): most numerical-precision/threshold/seed/model
defaults come out ``require`` because no in-repo citation for the specific
value exists today. Sourcing a defensible value, or overriding an edge-case
misclassification, is deferred to the per-module implementation sub-issues.

Usage:
    python tools/classify_renames_and_defaults.py --date 20260917
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FIELDS = [
    "source",
    "current_name",
    "proposed_name",
    "name_decision",
    "name_rationale",
    "module",
    "kind",
    "parameter",
    "current_default",
    "default_decision",
    "default_rationale",
    "migration_note",
]

# --- naming: controlled verb vocabulary (ADR-0028, "Naming convention") ----
# Every entry is the first token of a compliant name. Definitions live in the
# ADR itself; this set is what the classifier checks the first token against.
VERB_VOCAB = {
    "fit", "score", "predict", "simulate", "estimate", "validate", "check",
    "build", "render", "select", "assemble", "equate", "evaluate",
    "calibrate", "resolve", "load", "save", "normalize", "detect",
    "compute", "compare", "run", "analyze", "enumerate", "generate",
    "administer", "align", "audit", "classify", "compile", "describe",
    "diagnose", "draw", "execute", "expand", "export", "extract", "get",
    "govern", "link", "list", "migrate", "parse", "prepare", "project",
    "route", "smooth", "standardize", "count", "rotate",
}

# --- naming: eponym exceptions (ADR-0028 rule 1's named-procedure clause) --
# A callable named after the published statistic/algorithm/rating system it
# implements, where that name (not a verb paraphrase) is the term the
# psychometric/measurement literature and this repository's own accepted
# ADRs (0011, 0017, 0018, 0021, 0022, 0025) already use for it. Kept as-is:
# forcing a verb prefix here would make these harder to find, not easier,
# since callers search by the term they already know from the paper.
EPONYM_KEEP = {
    "a_stratified", "andersen_lr_test", "benjamini_hochberg", "bhapkar_mh",
    "bradley_terry_mm", "bratt_mm", "circular_triads", "cronbach_alpha",
    "delta_plot", "dimtest", "elo_rating", "elom_rating",
    "feldt_alpha_ci", "fide_rating", "finite_population_achieved_proportion",
    "finite_population_proportion_design", "finn_coefficient",
    "fleiss_kappa", "gauss_hermite_nodes", "gbt", "glb_fa",
    "glb_fa_from_data", "glicko2_rating", "glicko_rating", "gtheory_pi",
    "gtheory_pio", "guttman_lambdas", "hanson_brennan",
    "hanson_brennan_from_params", "hofstee", "icc", "ilsr_pairwise",
    "ilsr_rankings", "ilsr_top1", "infit_outfit", "k_index", "kendall_u",
    "kripp_alpha", "lee_classification", "light_kappa",
    "livingston_correlation", "livingston_k2", "livingston_lewis", "logit",
    "lsr_pairwise", "lsr_rankings", "lsr_top1", "m2", "m2_cmle_rasch",
    "m2_multigroup", "m2_multilevel", "m2_polytomous", "main", "maxwell_re",
    "minres_fa", "minres_fa_from_data", "neg_loglik_and_grad", "oakes_standard_errors",
    "omega_total_1f", "omega_total_1f_from_data", "owen_cat", "owen_update",
    "parallel_analysis", "phi_lambda", "raju_area", "rank_centrality",
    "residual_interaction_map", "robinson_a", "rudner_classification",
    "s_x2", "sibtest", "sigmoid", "softplus", "stephenson_rating",
    "stuart_maxwell_mh", "subkoviak_agreement", "sympson_hetter",
    "taylor_russell", "tenberge_mu", "thurstone_case_v", "velicer_map",
    "velicer_map_from_data", "vuong_nonnested", "wollack_omega",
    "woodruff_sawyer_normal", "woodruff_sawyer_sb",
}
EPONYM_RATIONALE = (
    "ADR-0028 rule 1 eponym exception: name is the published statistic's/"
    "algorithm's literature-standard term, not a generic object noun; a "
    "verb paraphrase would obscure the reference rather than clarify it."
)

# --- naming: known ADR-0028 violations -> corrected name -------------------
# Every non-verb-first, non-eponym name found in the 2026-09-17 inventory,
# hand-reviewed and assigned a verb from VERB_VOCAB (docs/adr/0028-*,
# "Naming convention"). Keyed by current_name; covers Python names only
# (PyO3 entry points are exempt, rule 6).
RENAME_MAP = {
    # polytomous: majority is a trailing scope qualifier. Rule 4: when an
    # object noun and a scope qualifier are both present, object precedes
    # scope (<verb>_<model>_<object>_<scope>), so polytomous stays LAST
    # even though the verb/object it follows changes per function.
    "polytomous_expected_response": "predict_expected_response_polytomous",
    "polytomous_category_probabilities": "predict_category_probabilities_polytomous",
    "polytomous_information_criteria": "compute_information_criteria_polytomous",
    "dif_polytomous_anchor_sets": "detect_dif_anchor_sets_polytomous",
    "dif_polytomous_purified": "detect_dif_polytomous_purified",
    "dif_polytomous": "detect_dif_polytomous",
    "information_polytomous": "compute_information_polytomous",
    "local_dependence_polytomous": "diagnose_local_dependence_polytomous",
    "item_fit_polytomous": "compute_item_fit_polytomous",
    "person_fit_polytomous": "compute_person_fit_polytomous",
    "u3_cutoff_polytomous": "compute_u3_cutoff_polytomous",
    "u3_person_fit_polytomous": "compute_u3_person_fit_polytomous",
    "expected_total_score_monotonicity": "check_expected_total_score_monotonicity",
    "bifactor_expected_total_score_monotonicity": "check_bifactor_expected_total_score_monotonicity",
    "focal_expected_total_score_monotonicity": "check_focal_expected_total_score_monotonicity",
    # bifactor: majority is verb-adjacent prefix, not trailing.
    "direct_enumeration_bifactor": "enumerate_bifactor_direct",
    "bifactor_lord_wingersky": "enumerate_bifactor_lord_wingersky",
    "bifactor_scoreability": "assess_bifactor_scoreability",
    "bifactor_scoreability_from_logit_slopes": "assess_bifactor_scoreability_from_logit_slopes",
    # DIF family: unify on the detect_dif_<method> prefix form.
    "mantel_haenszel_dif": "detect_dif_mantel_haenszel",
    "gmh_dif": "detect_dif_gmh",
    "breslow_day_dif": "detect_dif_breslow_day",
    "logistic_dif": "detect_dif_logistic",
    "mantel_haenszel_dif_purified": "detect_dif_mantel_haenszel_purified",
    "logistic_dif_purified": "detect_dif_logistic_purified",
    "mantel_smd_dif": "detect_dif_mantel_smd",
    "eb_mh_dif": "detect_dif_eb_mh",
    "dif_analysis": "detect_dif_summary",
    # verb-first violations (qualifier before verb).
    "cat_simulate_polytomous": "simulate_cat_polytomous",
    "cat_next_item": "select_cat_next_item",
    "ccat_select": "select_ccat",
    "ci_classify": "classify_ci",
    "sprt_classify": "classify_sprt",
    "epv_select": "select_epv",
    "kl_select": "select_kl",
    "irtree_expand": "expand_irtree",
    "loglinear_smooth": "smooth_loglinear",
    "two_stage_route": "route_two_stage",
    "two_stage_score": "score_two_stage",
    # CAT-administration family: unify on administer_<method> prefix,
    # matching the already-compliant administer_adaptive_test.
    "flexilevel_administer": "administer_flexilevel",
    "pyramidal_administer": "administer_pyramidal",
    "stradaptive_administer": "administer_stradaptive",
    # equating/linking family: unify on the equate_/link_ verb.
    "circle_arc_equate": "equate_circle_arc",
    "circle_arc_middle_anchor": "compute_circle_arc_middle_anchor",
    "nominal_weights_mean_equate": "equate_nominal_weights_mean",
    "composite_linking": "link_composite",
    "irt_link": "link_irt",
    # generic object nouns needing a verb (default "compute" unless a more
    # specific verb applies).
    "ability_standard_error": "compute_ability_standard_error",
    "adjusted_chi2_pairs": "compute_adjusted_chi2_pairs",
    "available_rotation_criteria": "list_available_rotation_criteria",
    "bank_information": "compute_bank_information",
    "canonical_generation_contract": "get_canonical_generation_contract",
    "category_logprobs": "compute_category_logprobs",
    "chi2_sf": "compute_chi2_sf",
    "confirmatory": "build_confirmatory_model",
    "exploratory": "build_exploratory_model",
    "dimensionality_diagnostics": "diagnose_dimensionality",
    "dimensionality_residuals": "compute_dimensionality_residuals",
    "empirical_reliability": "estimate_empirical_reliability",
    "separation_reliability": "estimate_separation_reliability",
    "enterprise_issue_evidence_references": "get_enterprise_issue_evidence_references",
    "equating_standard_errors": "compute_equating_standard_errors",
    "exact_value_csv": "render_exact_value_csv",
    "exact_value_json": "render_exact_value_json",
    "exact_value_text": "render_exact_value_text",
    "exact_value_disclosure": "validate_exact_value_disclosure",
    "fixed_item_calibration_diagnostics": "diagnose_fixed_item_calibration",
    "flexilevel_score_distribution": "score_flexilevel_distribution",
    "gdina_wald_selection": "compute_gdina_wald_selection",
    "gpcm_node_gradient": "compute_gpcm_node_gradient",
    "grm_category_logprobs": "compute_grm_category_logprobs",
    "k_variants": "list_k_variants",
    "kl_information": "compute_kl_information",
    "ksirt_analysis": "analyze_ksirt",
    "linear_predictor": "compute_linear_predictor",
    "mean_pairwise_cor": "compute_mean_pairwise_cor",
    "mean_pairwise_rho": "compute_mean_pairwise_rho",
    "metrics_rating": "compute_metrics_rating",
    "model_flags": "get_model_flags",
    "mokken_analysis": "analyze_mokken",
    "n_cohen_kappa": "compute_n_cohen_kappa",
    "n_dims_of": "count_dimensions_of",
    "observed_information": "compute_observed_information",
    "ordered_column_names": "get_ordered_column_names",
    "paired_rating_range_evidence": "compute_paired_rating_range_evidence",
    "person_fit": "compute_person_fit",
    "person_fit_np": "compute_person_fit_np",
    "person_fit_resampling": "compute_person_fit_resampling",
    "plausible_values": "draw_plausible_values",
    "rag_evidence_regime_limitations": "get_rag_evidence_regime_limitations",
    "rater_bias": "estimate_rater_bias",
    "recovery_report": "build_recovery_report",
    "residual_item_fit": "compute_residual_item_fit",
    "response_process_dimensionality_diagnostics": "diagnose_response_process_dimensionality",
    "response_process_fit_diagnostics": "diagnose_response_process_fit",
    "rotation_criterion_value_gradient": "compute_rotation_criterion_value_gradient",
    "rt_person_fit": "compute_rt_person_fit",
    "second_order_test": "evaluate_second_order_test",
    "selection_utility": "compute_selection_utility",
    "serving_prior": "get_serving_prior",
    "standard_errors_from_vcov": "compute_standard_errors_from_vcov",
    "subscore_analysis": "analyze_subscore",
    "tcc_drift": "compute_tcc_drift",
    "vcov_from_hessian": "compute_vcov_from_hessian",
    "weighted_contextual_effect": "compute_weighted_contextual_effect",
    "item_information": "compute_item_information",
    "item_information_matrix": "compute_item_information_matrix",
}

# --- defaults: parameter-name -> policy family ------------------------------
QUADRATURE_RE = re.compile(
    r"^(q_theta|q_xi|q_u|q_gamma|n_quad|n_nodes|xi_points|m2_q_theta|m2_q_u|"
    r"m2_q_xi|nevalpoints|node_dims)$"
)
CONVERGENCE_RE = re.compile(
    r"^(max_iter|tol|tolerance|max_cycles|max_line_search|m_steps|mh_steps|"
    r"n_iterations|burn_in|target_accept|basin_tolerance|omega_tol|"
    r"sig2_stop|hessian_step|eps|eps_distance|epsilon)$"
)
REPLICATES_RE = re.compile(
    r"^(n_rep|n_replicates|n_boot|n_draws|n_starts|n_sample_splits|k_folds|"
    r"init_games)$"
)
SEED_RE = re.compile(r"seed")
THRESHOLD_RE = re.compile(
    r"^(alpha|alpha_level|fdr_q|se_threshold|flag_threshold|min_expected|"
    r"min_discrimination|min_effect|person_flag_threshold|"
    r"itemfit_penalty_weight|msq_band|isolation_z|sx2_min_effect|"
    r"zero_tolerance|max_rounds|min_anchor_items|min_flags_to_remove|j_min|"
    r"ci_level|conf_level|centile|z_fast)$"
)
MODEL_RE = re.compile(r"^model$")

PARAM_RE = re.compile(r"(\w+)\s*=\s*((?:\"[^\"]*\")|(?:'[^']*')|[^,]+)")


def _classify_default(param_name: str) -> tuple[str, str] | None:
    """Return (decision, rationale) for a defaulted parameter, or None if
    the parameter is out of ADR-0028's scope (keeps its default silently)."""
    if QUADRATURE_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 1: numerical-precision (quadrature node count) "
            "control; no in-repo citation for this value and/or below the "
            "AGENTS.md #1929 floor of 121 -> required argument.",
        )
    if CONVERGENCE_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 1: iteration/convergence precision control; no "
            "documented convergence-criterion source -> required argument.",
        )
    if REPLICATES_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 1: replicate/bootstrap/draw count; no "
            "Monte-Carlo-error source cited -> required argument.",
        )
    if SEED_RE.search(param_name):
        return (
            "require",
            "ADR-0028 rule 3: stochastic routines must not ship a default "
            "seed -> required argument.",
        )
    if MODEL_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 4: function is model-generic (not model-specific "
            "by name) -> model choice has no default.",
        )
    if THRESHOLD_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 2: decision threshold / flag cutoff / stopping "
            "rule; no APA 7th source cited for this specific value -> "
            "required argument.",
        )
    return None


def _parse_params(parameters: str) -> list[tuple[str, str | None]]:
    """Return [(name, default_repr_or_None), ...] from an inventory row's
    ``parameters`` column."""
    out: list[tuple[str, str | None]] = []
    depth = 0
    current = ""
    parts = []
    for ch in parameters:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    for p in parts:
        if not p or p.startswith("*"):
            continue
        if "=" in p:
            name, default = p.split("=", 1)
            out.append((name.strip(), default.strip()))
        else:
            out.append((p.strip(), None))
    return out


def build_rows(inventory_rows: list[dict]) -> list[dict]:
    out_rows: list[dict] = []
    for r in inventory_rows:
        current_name = r["current_name"]
        module = r["module"]
        source = r["source"]
        kind = r["kind"]
        if source == "pyo3":
            proposed_name = current_name
            name_decision = "keep"
            name_rationale = (
                "ADR-0028: PyO3 entry points are a private implementation "
                "detail, exempt from the public naming rules."
            )
        elif kind == "class":
            proposed_name = "keep"
            name_decision = "keep"
            name_rationale = (
                "ADR-0028 rule 1 (verb-first) governs free functions; "
                "classes follow the existing PascalCase noun convention, "
                "which this ADR does not change."
            )
        elif current_name in EPONYM_KEEP:
            proposed_name = "keep"
            name_decision = "keep"
            name_rationale = EPONYM_RATIONALE
        elif current_name.split("_")[0] in VERB_VOCAB:
            proposed_name = "keep"
            name_decision = "keep"
            name_rationale = (
                "ADR-0028 rule 1: first token is in the controlled verb "
                "vocabulary -> already compliant."
            )
        elif current_name in RENAME_MAP:
            proposed_name = RENAME_MAP[current_name]
            name_decision = "rename"
            name_rationale = (
                "ADR-0028 naming convention: verb-first, model/family token "
                "as a verb-adjacent prefix, scope qualifier (data shape) as "
                "a trailing suffix; hand-reviewed against the 2026-09-17 "
                "inventory (not a bare eponym, see EPONYM_KEEP)."
            )
        else:
            # Safety net: should be empty against the 2026-09-17 inventory
            # (every violation was hand-reviewed into RENAME_MAP or
            # EPONYM_KEEP); a future regeneration that finds a genuinely new
            # name here needs the same hand review, not a guessed default.
            proposed_name = "NEEDS-MANUAL-REVIEW"
            name_decision = "review"
            name_rationale = (
                "ADR-0028: first token not in the verb vocabulary and not "
                "in EPONYM_KEEP or RENAME_MAP -- new name since the last "
                "hand review, needs a human naming decision before this row "
                "can be `keep` or `rename`."
            )

        params = _parse_params(r.get("parameters", ""))
        defaulted = [(n, d) for n, d in params if d is not None]

        if not defaulted:
            out_rows.append(
                {
                    "source": source,
                    "current_name": current_name,
                    "proposed_name": proposed_name,
                    "name_decision": name_decision,
                    "name_rationale": name_rationale,
                    "module": module,
                    "kind": kind,
                    "parameter": "",
                    "current_default": "",
                    "default_decision": "n/a",
                    "default_rationale": "No defaulted parameters.",
                    "migration_note": "",
                }
            )
            continue

        for pname, pdefault in defaulted:
            classification = _classify_default(pname)
            if classification is None:
                decision, rationale = (
                    "keep",
                    "Out of ADR-0028 policy scope (not a precision control, "
                    "decision threshold, seed, or model-choice parameter).",
                )
            else:
                decision, rationale = classification
            migration_note = (
                (
                    f"Deprecation: keep {pname}={pdefault} accepted only when "
                    "explicitly passed for one minor release "
                    "(DeprecationWarning), then remove; changelog Changed "
                    "entry."
                )
                if decision in ("require", "change")
                else ""
            )
            out_rows.append(
                {
                    "source": source,
                    "current_name": current_name,
                    "proposed_name": proposed_name,
                    "name_decision": name_decision,
                    "name_rationale": name_rationale,
                    "module": module,
                    "kind": kind,
                    "parameter": pname,
                    "current_default": pdefault,
                    "default_decision": decision,
                    "default_rationale": rationale,
                    "migration_note": migration_note,
                }
            )
    out_rows.sort(key=lambda r: (r["module"], r["current_name"], r["parameter"]))
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    in_path = REPO_ROOT / "docs" / "api" / f"inventory-{args.date}.csv"
    out_path = REPO_ROOT / "docs" / "api" / f"renames-and-defaults-{args.date}.csv"

    with in_path.open(newline="") as fh:
        inventory_rows = list(csv.DictReader(fh))

    out_rows = build_rows(inventory_rows)

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
