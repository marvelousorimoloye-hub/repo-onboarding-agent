"""
Central configuration for the codebase onboarding / doc-drift agent swarm.

Phase 0 — this file is fully implemented (not a skeleton). Everything else
in the project reads model/DB config from here rather than touching
environment variables directly.
"""
import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ if present; no-op if it doesn't exist


class ModelProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"


class AgentRole(str, Enum):
    """
    Every agent in the swarm maps to one of these roles, which in turn maps
    to a model provider. See MODEL_ROLE_MAP below.
    """
    ROUTER = "router"
    RETRIEVAL = "retrieval"
    DRIFT_DETECTOR = "drift_detector"
    TRACE_AGENT = "trace_agent"
    ARCHITECTURE_SYNTHESIZER = "architecture_synthesizer"
    VERIFIER = "verifier"
    ANSWER_SYNTHESIZER = "answer_synthesizer"


# --- Role -> provider mapping (dev) -----------------------------------------
# Fast/high-frequency roles -> Groq. Careful-reasoning roles -> Gemini.
# Swap this map's values (not the keys, not calling code) to move to
# stronger prod models later.
MODEL_ROLE_MAP: dict[AgentRole, ModelProvider] = {
    AgentRole.ROUTER: ModelProvider.GROQ,
    AgentRole.RETRIEVAL: ModelProvider.GROQ,
    AgentRole.DRIFT_DETECTOR: ModelProvider.GROQ,
    AgentRole.TRACE_AGENT: ModelProvider.GEMINI,
    AgentRole.ARCHITECTURE_SYNTHESIZER: ModelProvider.GEMINI,
    AgentRole.VERIFIER: ModelProvider.GEMINI,
    AgentRole.ANSWER_SYNTHESIZER: ModelProvider.GEMINI,
}

# --- Concrete model names per provider (dev) --------------------------------
# Centralized so a prod swap is a one-line change per provider, not a
# find-and-replace across the codebase.
DEV_MODEL_NAMES: dict[ModelProvider, str] = {
    ModelProvider.GROQ: os.getenv("GROQ_DEV_MODEL", "openai/gpt-oss-20b"),

    ModelProvider.GEMINI: os.getenv("GEMINI_DEV_MODEL", "gemini-2.5-flash"),
}

# Embedding model for the Semantic Indexer (Phase 2). Gemini, not Groq —
# Groq doesn't currently offer an embeddings endpoint. Kept separate from
# MODEL_ROLE_MAP since embedding isn't a chat-completion "agent role".
# text-embedding-004 (legacy) has been superseded — gemini-embedding-001 is
# the current GA model as of this writing. Its NATIVE output is 3072-dim;
# EMBEDDING_DIM below (768, matching db/schema.sql's vector(768) column) is
# requested explicitly via output_dimensionality in the embed_content call
# (see ingestion/semantic_indexer.py), not the model's default.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # must match db/schema.sql's vector(768)


@dataclass(frozen=True)
class DBSettings:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "onboarding_agent")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class APIKeys:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")


DB = DBSettings()
KEYS = APIKeys()

# Where the Repo Registry clones repos to on disk before RepoMapper reads them.
REPOS_CACHE_DIR = os.getenv("REPOS_CACHE_DIR", "./repo_cache")

# For PR description ingestion (Phase 2, Semantic Indexer). Optional —
# unauthenticated GitHub API calls work but are capped at 60/hour; a
# token raises that to 5000/hour and is required for private repos.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def get_model_provider(role: AgentRole) -> ModelProvider:
    return MODEL_ROLE_MAP[role]


def get_model_name(role: AgentRole) -> str:
    return DEV_MODEL_NAMES[get_model_provider(role)]