from functools import lru_cache
from google import genai
from docent.config.settings import settings


class GeminiClient:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        self._embedding_model = settings.gemini_embedding_model

    def generate(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return resp.text.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.models.embed_content(
            model=self._embedding_model,
            contents=texts,
        )
        return [e.values for e in resp.embeddings]


@lru_cache
def get_llm() -> GeminiClient:
    return GeminiClient()