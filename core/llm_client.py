"""
core/llm_client.py
Wraps the Ollama REST API for local LLM inference.

Supports:
- Single-turn generation (generate)
- Health check (is_available)
- Model listing (list_models)
"""

import json
import urllib.request
import urllib.error
from utils.logger import get_logger
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, SYSTEM_PROMPT

log = get_logger("LLMClient")


class LLMClient:
    """
    Thin wrapper around the Ollama HTTP API.
    No external SDK dependency — uses only stdlib urllib.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str    = OLLAMA_MODEL,
        timeout: int  = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout
        log.info(f"LLMClient → model='{self.model}' | endpoint={self.base_url}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        """
        Send a prompt to Ollama and return the response text.

        Args:
            prompt:  The user prompt
            system:  System instruction (defaults to SYSTEM_PROMPT from config)

        Returns:
            Generated text string
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.3,    # Low temperature = more factual, deterministic
                "num_predict": 2048,   # Max tokens in response
            },
        }

        try:
            response_data = self._post("/api/generate", payload)
            return response_data.get("response", "").strip()

        except ConnectionRefusedError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            log.error(f"LLM generation error: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama is running and the configured model is available."""
        try:
            data = self._get("/api/tags")
            models = [m["name"] for m in data.get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                log.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. "
                    f"Pull it with: ollama pull {self.model}"
                )
            return available
        except Exception as e:
            log.error(f"Ollama not reachable: {e}")
            return False

    def list_models(self) -> list[str]:
        """Return list of locally available Ollama model names."""
        try:
            data = self._get("/api/tags")
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Internal HTTP helpers ──────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        url  = self.base_url + path
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
