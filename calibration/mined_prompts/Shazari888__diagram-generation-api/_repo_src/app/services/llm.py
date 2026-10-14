from openai import AsyncOpenAI
import logging

from fastapi import HTTPException

from app.config import settings

log = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You generate diagram source code only.
Return raw {diagram_type} syntax with no markdown fences or explanation."""

FALLBACK_BY_TYPE = {
    "mermaid": "graph TD; A[Start] --> B[Login]; B --> C{Success?}; C -->|Yes| D[Dashboard]; C -->|No| E[Retry]",
    "d2": "Start: Start\nLogin: Login\nSuccess: Success?\nDashboard: Dashboard\nRetry: Retry\nStart -> Login\nLogin -> Success\nSuccess -> Dashboard: Yes\nSuccess -> Retry: No",
    "plantuml": "@startuml\nstart\n:Login;\nif (Success?) then (yes)\n  :Dashboard;\nelse (no)\n  :Retry;\nendif\nstop\n@enduml",
    "graphviz": 'digraph G { A [label="Start"]; B [label="Login"]; C [label="Success?"]; D [label="Dashboard"]; E [label="Retry"]; A -> B; B -> C; C -> D [label="Yes"]; C -> E [label="No"]; }',
}


async def generate_source(prompt: str, diagram_type: str) -> str:
    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(diagram_type=diagram_type)},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty diagram source")
        return content.strip()
    except Exception as exc:
        if settings.use_llm_fallback:
            log.warning(
                "OpenAI call failed (%s). Using fallback diagram (use_llm_fallback=true).",
                type(exc).__name__,
            )
            return FALLBACK_BY_TYPE.get(diagram_type.lower(), FALLBACK_BY_TYPE["mermaid"])
        log.error("OpenAI call failed (%s): %s", type(exc).__name__, str(exc)[:200])
        raise HTTPException(
            status_code=503,
            detail=f"Diagram generation unavailable: upstream LLM error ({type(exc).__name__})",
        ) from exc