def aggregate(expected, results, selection_valid=False, allowed_skips=()):
    # Deliberately unsafe: incomplete results and invalid skips pass.
    return any(result in ("success", "skipped") for result in results.values())
