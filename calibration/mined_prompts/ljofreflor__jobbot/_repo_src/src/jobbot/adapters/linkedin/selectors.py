"""LinkedIn selector hints."""

LINKEDIN_FEED = "https://www.linkedin.com/feed/"
LINKEDIN_LOGIN = "https://www.linkedin.com/login"
LINKEDIN_ME = "https://www.linkedin.com/in/me/"

# Post card control menu ("…"): the only place LinkedIn still hands over a
# post's own URL, via "Copy link to post" → clipboard.
POST_CONTROL_MENU = (
    '[aria-label^="Abrir el menú de controles"], '
    '[aria-label^="Open control menu"], '
    '[aria-label*="control menu for"]'
)
POST_MENU_ITEM = '[role="menuitem"], [role="menu"] [role="button"]'
COPY_LINK_LABELS = ("copiar enlace", "copy link")
