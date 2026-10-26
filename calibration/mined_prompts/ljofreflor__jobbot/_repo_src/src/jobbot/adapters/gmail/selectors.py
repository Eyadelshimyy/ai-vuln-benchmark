"""Gmail web selectors. SEND_BUTTON is here for assertions only — never clicked."""

from __future__ import annotations

GMAIL_HOME = "https://mail.google.com/mail/u/0/#inbox"

# `?view=cm&fs=1` opens the standalone (full-window) compose, which is NOT a
# role="dialog". The subject input exists in both the standalone and the popup
# compose, so it is the reliable "compose is ready" signal.
COMPOSE_READY = 'input[name="subjectbox"]'
FILE_INPUT = 'input[type="file"][name="Filedata"], input[type="file"]'
ATTACH_BUTTON = (
    '[command="Files"], [aria-label*="Attach files"], [aria-label*="Adjuntar archivos"]'
)
SEND_BUTTON = '[role="button"][aria-label^="Send"], [role="button"][aria-label^="Enviar"]'
DISCARD_BUTTON = (
    '[aria-label*="Discard draft"], [aria-label*="Descartar borrador"], [command="Files"] ~ '
    '[aria-label*="Descartar"]'
)
