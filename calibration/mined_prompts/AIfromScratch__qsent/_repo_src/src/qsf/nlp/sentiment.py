"""
HuggingFace FinBERT sentiment model.
"""
import os
import time

import requests

from qsf.common.logging import get_current_trace, get_logger

logger = get_logger(__name__)

FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}
HF_REQUEST_TIMEOUT = 10  # seconds — applies to health check and per-item scoring requests


class FinBERTModel:
    def score(self, texts: list[str]) -> list[float | None]:
        """Score sentiment for each text individually.

        The HuggingFace Inference API for FinBERT only processes the first item
        when given a batch, so we send one text per request.

        Returns a list of the same length as the input. Failed items are None
        rather than omitted, so callers can zip safely without truncation.
        """
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        scores: list[float | None] = [None] * len(texts)
        failed = 0
        trace = get_current_trace()

        logger.info("FinBERTModel.score: scoring %d items via HuggingFace API (1 request per item)", len(texts))
        if not texts:
            return scores
        try:
            t0 = time.monotonic()
            probe = requests.post(FINBERT_URL, headers=headers, json={"inputs": "test"}, timeout=HF_REQUEST_TIMEOUT)
            elapsed = time.monotonic() - t0
            logger.info("FinBERTModel.score: HuggingFace health check — HTTP %d in %.2fs", probe.status_code, elapsed)
        except Exception as e:
            logger.warning("FinBERTModel.score: HuggingFace health check failed — %s: %s", type(e).__name__, e)
        for idx, text in enumerate(texts):
            if idx % 10 == 0:
                logger.info("FinBERTModel.score: progress %d/%d", idx, len(texts))
            try:
                current_text = text
                while True:
                    response = requests.post(FINBERT_URL, headers=headers, json={"inputs": current_text}, timeout=HF_REQUEST_TIMEOUT)
                    if response.status_code == 400 and "size of tensor" in response.text:
                        trimmed_len = len(current_text) - 200
                        if trimmed_len <= 0:
                            response.raise_for_status()
                        logger.warning(
                            "FinBERTModel.score: item %d/%d exceeded token limit (%d chars), trimming to %d chars",
                            idx + 1, len(texts), len(current_text), trimmed_len,
                        )
                        current_text = current_text[:trimmed_len]
                        continue
                    response.raise_for_status()
                    break
                result = response.json()
                if not isinstance(result, list) or not result:
                    logger.warning(
                        "FinBERTModel.score: item %d/%d returned unexpected response: %s",
                        idx + 1, len(texts), str(result)[:200],
                    )
                    if trace: trace.generation(name="finbert", model="ProsusAI/finbert", input=current_text[:200], level="WARNING", status_message="unexpected_response")
                    failed += 1
                    continue
                # Single-text requests return [[{label_dicts}]] — unwrap the outer list
                label_scores = result[0] if isinstance(result[0], list) else result
                top = max(label_scores, key=lambda x: x["score"])
                score = SENTIMENT_MAP.get(top["label"].lower(), 0) * top["score"]
                logger.info(
                    "FinBERTModel.score: item %d/%d — %s (%.3f)",
                    idx + 1, len(texts), top["label"].lower(), score,
                )
                if trace: trace.generation(name="finbert", model="ProsusAI/finbert", input=current_text[:200], output=str(round(score, 4)))
                scores[idx] = score
            except requests.HTTPError as e:
                logger.warning(
                    "FinBERTModel.score: item %d/%d failed — HTTP %s: %s",
                    idx + 1, len(texts), e.response.status_code, e.response.text[:200],
                )
                if trace: trace.generation(name="finbert", model="ProsusAI/finbert", input=text[:200], level="ERROR", status_message=f"HTTP {e.response.status_code}")
                failed += 1
            except Exception as e:
                logger.warning(
                    "FinBERTModel.score: item %d/%d failed — %s: %s",
                    idx + 1, len(texts), type(e).__name__, e,
                )
                if trace: trace.generation(name="finbert", model="ProsusAI/finbert", input=text[:200], level="ERROR", status_message=f"{type(e).__name__}: {e}")
                failed += 1

        if failed:
            logger.warning(
                "FinBERTModel.score: %d/%d items failed to score",
                failed, len(texts),
            )

        return scores
