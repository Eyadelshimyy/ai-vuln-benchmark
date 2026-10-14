"""`fslm` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from .definition import MachineDefinition
from .kernel import FSLMKernel
from .loading import load_machine_definition
from .models import FSLMEvent, MachineSnapshot
from .rendering import ColorMode, render_snapshot, render_step, render_validation
from .visualization import to_mermaid

OutputFormat = Literal["json", "rich"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fslm",
        description="Run and inspect ellements finite-state linguistic machines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a Python-defined machine.")
    validate.add_argument("spec")
    validate.add_argument("--bindings", action="append", default=None)
    _add_human_output_options(validate)
    validate.set_defaults(handler=_validate)

    mermaid = sub.add_parser("mermaid", help="Render a spec as Mermaid.")
    mermaid.add_argument("spec")
    mermaid.add_argument("--bindings", action="append", default=None)
    mermaid.set_defaults(handler=_mermaid)

    init_state = sub.add_parser("init-state", help="Create an initial snapshot.")
    init_state.add_argument("spec")
    init_state.add_argument("--bindings", action="append", default=None)
    init_state.add_argument("--machine-id", default=None)
    init_state.add_argument("--seed", type=int, default=None)
    init_state.add_argument("--state-dir", type=Path, default=None)
    _add_human_output_options(init_state)
    init_state.set_defaults(handler=_init_state)

    inspect = sub.add_parser("inspect", help="Print a snapshot JSON file.")
    inspect.add_argument("snapshot", type=Path)
    inspect.add_argument("--spec", default=None, help="Optional machine spec for state objectives.")
    _add_human_output_options(inspect)
    inspect.set_defaults(handler=_inspect)

    step = sub.add_parser("step", help="Run one machine step.")
    step.add_argument("spec")
    step.add_argument("--bindings", action="append", default=None)
    step.add_argument("--snapshot", type=Path)
    step.add_argument("--state-dir", type=Path, default=None)
    step.add_argument("--event", type=Path, required=True)
    _add_human_output_options(step)
    step.set_defaults(handler=_step)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler: Callable[[argparse.Namespace], int] = args.handler
        return handler(args)
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1


def _validate(args: argparse.Namespace) -> int:
    spec = _load_definition(args).spec
    if _output_format(args) == "rich":
        render_validation(spec, color=_color_mode(args))
    else:
        print(json.dumps({"ok": True, "machine": spec.name}, sort_keys=True))
    return 0


def _mermaid(args: argparse.Namespace) -> int:
    print(to_mermaid(_load_definition(args).spec), end="")
    return 0


def _init_state(args: argparse.Namespace) -> int:
    spec = _load_definition(args).spec
    snapshot = spec.initial_snapshot(
        machine_id=args.machine_id,
        random_seed=args.seed,
    )
    if args.state_dir is not None:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        (args.state_dir / "snapshot.json").write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if _output_format(args) == "rich":
        render_snapshot(
            snapshot,
            spec=spec,
            title="FSLM Initial State",
            color=_color_mode(args),
        )
    else:
        print(json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    snapshot = MachineSnapshot.model_validate(
        json.loads(args.snapshot.read_text(encoding="utf-8"))
    )
    if _output_format(args) == "rich":
        spec = _load_definition(args).spec if args.spec is not None else None
        render_snapshot(snapshot, spec=spec, color=_color_mode(args))
    else:
        print(json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _step(args: argparse.Namespace) -> int:
    definition = _load_definition(args)
    snapshot_path = args.snapshot
    if snapshot_path is None and args.state_dir is not None:
        snapshot_path = args.state_dir / "snapshot.json"
    if snapshot_path is None:
        raise ValueError("--snapshot or --state-dir is required")
    snapshot = MachineSnapshot.model_validate(json.loads(snapshot_path.read_text("utf-8")))
    event = FSLMEvent.model_validate(json.loads(args.event.read_text(encoding="utf-8")))
    result = asyncio.run(FSLMKernel(definition).step(snapshot, event))
    if args.state_dir is not None:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        (args.state_dir / "snapshot.json").write_text(
            json.dumps(
                result.new_snapshot.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if _output_format(args) == "rich":
        render_step(
            result,
            definition=definition,
            event=event,
            state_dir=args.state_dir,
            color=_color_mode(args),
        )
    else:
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _add_human_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("json", "rich"),
        default="json",
        help="Output format. JSON is stable for automation; rich is designed for humans.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Color handling for rich output.",
    )


def _output_format(args: argparse.Namespace) -> OutputFormat:
    value = args.format
    if value not in {"json", "rich"}:
        raise ValueError(f"unsupported output format: {value}")
    return cast(OutputFormat, value)


def _color_mode(args: argparse.Namespace) -> ColorMode:
    value = args.color
    if value not in {"auto", "always", "never"}:
        raise ValueError(f"unsupported color mode: {value}")
    return cast(ColorMode, value)


def _load_definition(args: argparse.Namespace) -> MachineDefinition:
    bindings = getattr(args, "bindings", None)
    return load_machine_definition(args.spec, binding_modules=list(bindings or []))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
