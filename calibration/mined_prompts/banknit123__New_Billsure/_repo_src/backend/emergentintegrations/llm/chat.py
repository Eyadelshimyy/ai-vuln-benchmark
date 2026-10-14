"""Local compatibility shim for emergentintegrations.llm.chat.

The real `emergentintegrations` package (Emergent's own LLM/payments
wrapper) has been removed from PyPI entirely -- it is no longer
installable anywhere, which blocked running this backend outside
Emergent's own infrastructure at all. Both call sites in server.py that
use LlmChat/UserMessage/ImageContent (extract_bill_with_vision, and the
AI-insights block in the analytics endpoint) already check
`if EMERGENT_LLM_KEY:` before doing anything with these classes, and that
env var is unset by default -- so this shim only needs to be IMPORTABLE
for the app to run, not functionally complete.

If you want the AI bill-scanning / spending-insights features working
again, wire LlmChat.send_message() below up to a real provider you hold a
key for (OpenAI, Google Gemini, Anthropic) -- both call sites are already
isolated behind this one method, so that's the only place a real
implementation needs to go.
"""


class UserMessage:
    def __init__(self, text: str, file_contents: list = None):
        self.text = text
        self.file_contents = file_contents or []


class ImageContent:
    def __init__(self, image_base64: str):
        self.image_base64 = image_base64


class LlmChat:
    def __init__(self, api_key: str, session_id: str, system_message: str = ""):
        self.api_key = api_key
        self.session_id = session_id
        self.system_message = system_message
        self._provider = None
        self._model = None

    def with_model(self, provider: str, model: str) -> "LlmChat":
        self._provider = provider
        self._model = model
        return self

    async def send_message(self, message: UserMessage) -> str:
        raise NotImplementedError(
            "emergentintegrations.llm.chat is a local compatibility shim with no "
            "real provider wired up. Both callers check EMERGENT_LLM_KEY before "
            "reaching this point, so seeing this error means EMERGENT_LLM_KEY is "
            "set but LlmChat.send_message() was never connected to a real LLM "
            "provider -- see this module's docstring."
        )
