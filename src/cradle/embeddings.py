from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "mlx-community/embeddinggemma-300m-bf16"
DEFAULT_QUERY_TASK = "code retrieval"


class TextEmbedder(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> np.ndarray:
        ...


@dataclass(slots=True)
class MLXEmbedder:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    truncate_dim: int | None = None
    dimension: int = field(init=False)
    _model: object = field(init=False, repr=False)
    _tokenizer: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from mlx_embeddings import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-embeddings is required for semantic indexing and search on Apple Silicon. "
                "Install cradle with its MLX semantic dependency first."
            ) from exc

        self._model, self._tokenizer = load(self.model_name)
        probe = self._encode_batch([format_document_text("dimension probe")])
        base_dimension = int(probe.shape[1])
        if self.truncate_dim is not None and (self.truncate_dim <= 0 or self.truncate_dim > base_dimension):
            raise ValueError(f"truncate_dim must be between 1 and {base_dimension}, got {self.truncate_dim}")
        self.dimension = self.truncate_dim or int(base_dimension)

    def encode(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> np.ndarray:
        payload = list(texts)
        if not payload:
            return np.zeros((0, self.dimension), dtype=np.float32)

        batches: list[np.ndarray] = []
        total_batches = max(1, (len(payload) + self.batch_size - 1) // self.batch_size)
        for batch_index, start in enumerate(range(0, len(payload), self.batch_size), start=1):
            batch = payload[start : start + self.batch_size]
            batches.append(self._encode_batch(batch))
            if show_progress_bar and total_batches > 1:
                logger.info("embedded batch %s/%s with MLX", batch_index, total_batches)
        embeddings = np.concatenate(batches, axis=0)
        return truncate_embeddings(embeddings, self.truncate_dim)

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self._tokenizer(list(texts), padding=True, truncation=True, return_tensors="mlx")
        output = self._model(encoded["input_ids"], encoded["attention_mask"])
        return np.asarray(output.text_embeds, dtype=np.float32)


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    model_name: str = "google/embeddinggemma-300m"
    batch_size: int = 32
    truncate_dim: int | None = None
    device: str | None = None
    dimension: int = field(init=False)
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install cradle[cpu] to use the non-MLX embedding backend."
            ) from exc

        kwargs: dict[str, object] = {}
        if self.device is not None:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        base_dimension = self._model.get_sentence_embedding_dimension()
        if base_dimension is None:
            raise RuntimeError(f"Unable to determine embedding dimension for model {self.model_name!r}")
        if self.truncate_dim is not None and (self.truncate_dim <= 0 or self.truncate_dim > int(base_dimension)):
            raise ValueError(f"truncate_dim must be between 1 and {base_dimension}, got {self.truncate_dim}")
        self.dimension = self.truncate_dim or int(base_dimension)

    def encode(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> np.ndarray:
        payload = list(texts)
        if not payload:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            payload,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        return truncate_embeddings(embeddings, self.truncate_dim)


def build_text_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    batch_size: int = 32,
    truncate_dim: int | None = None,
    backend: str = "auto",
) -> TextEmbedder:
    resolved_backend = backend.lower()
    if resolved_backend not in {"auto", "mlx", "sentence-transformers"}:
        raise ValueError(f"Unsupported embedding backend: {backend}")

    if resolved_backend in {"auto", "mlx"}:
        try:
            return MLXEmbedder(model_name=model_name, batch_size=batch_size, truncate_dim=truncate_dim)
        except RuntimeError:
            if resolved_backend == "mlx":
                raise
            logger.info("MLX embedder unavailable for %s; trying sentence-transformers fallback", model_name)

    return SentenceTransformerEmbedder(model_name=model_name, batch_size=batch_size, truncate_dim=truncate_dim)


def format_query_text(query: str, *, task: str = DEFAULT_QUERY_TASK) -> str:
    return f"task: {task} | query: {query.strip()}"


def format_document_text(text: str, *, title: str | None = None) -> str:
    resolved_title = title.strip() if title and title.strip() else "none"
    return f"title: {resolved_title} | text: {text.strip()}"


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def truncate_embeddings(embeddings: np.ndarray, truncate_dim: int | None) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if truncate_dim is None:
        return normalize_embeddings(matrix)
    if truncate_dim <= 0 or truncate_dim > matrix.shape[1]:
        raise ValueError(f"truncate_dim must be between 1 and {matrix.shape[1]}, got {truncate_dim}")
    return normalize_embeddings(matrix[:, :truncate_dim])