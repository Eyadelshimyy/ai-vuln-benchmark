import argparse
import json
from pathlib import Path

import rox_core_domains
from rox_dox.features import build_feature_map, build_feature_maps
from rox_dox.schema import extract_tables

DEFAULT_REPO = Path("/home/ubuntu/repos/rox-core")
DEFAULT_COMMIT = Path(__file__).with_name("COMMIT").read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a commit-pinned rox-core feature map."
    )
    parser.add_argument("--domain", default="all")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    tables = extract_tables(args.repo.resolve(), args.commit)
    domain_tables = rox_core_domains.assign(sorted(tables))
    if args.domain != "all" and args.domain not in domain_tables:
        raise SystemExit(f"unknown domain: {args.domain}")
    if args.domain == "all":
        feature_maps = build_feature_maps(
            args.repo.resolve(),
            args.commit,
            domain_tables,
            threshold=args.threshold,
        )
    else:
        feature_maps = {
            args.domain: build_feature_map(
                args.repo.resolve(),
                args.commit,
                domain_tables,
                args.domain,
                threshold=args.threshold,
            )
        }

    output_dir = Path(__file__).resolve().parents[2] / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.domain == "all":
        for output_path in output_dir.glob("*.json"):
            if output_path.stem != "names" and output_path.stem not in domain_tables:
                output_path.unlink()
    for domain, feature_map in feature_maps.items():
        output_path = output_dir / f"{domain}.json"
        output_path.write_text(
            json.dumps(
                feature_map.model_dump(exclude_none=True),
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
