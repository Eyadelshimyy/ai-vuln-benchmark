"""Exit codes used by the JobBot CLI."""

SUCCESS = 0
GENERIC_FAILURE = 1
VALIDATION_FAILURE = 2
AUTH_REQUIRED = 3
MANUAL_CHALLENGE = 4
UI_CHANGED = 5
# `jobbot update --check`: a newer commit exists on the tracked ref.
UPDATE_AVAILABLE = 10
# typer turns KeyboardInterrupt into Exit(130) (128 + SIGINT).
USER_CANCEL = 130
