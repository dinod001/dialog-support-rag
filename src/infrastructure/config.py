import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class Config:
    """
    Central configuration class.
    Loads secrets from .env and YAML config files from /config.
    """

    def __init__(self) -> None:
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.config_dir = os.path.join(self.project_root, "config")

        env_path = os.path.join(self.project_root, ".env")
        load_dotenv(env_path)

        # --- ENV PROPERTIES ---
        self.env = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
            "QDRANT_URL": os.getenv("QDRANT_URL"),
            "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME"),
            "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
            "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "LANGFUSE_BASE_URL": os.getenv("LANGFUSE_BASE_URL"),
        }

        # Backward-compatible direct attributes.
        self.OPENAI_API_KEY = self.env["OPENAI_API_KEY"]
        self.QDRANT_API_KEY = self.env["QDRANT_API_KEY"]
        self.QDRANT_URL = self.env["QDRANT_URL"]
        self.QDRANT_COLLECTION_NAME = self.env["QDRANT_COLLECTION_NAME"]
        self.LANGFUSE_SECRET_KEY = self.env["LANGFUSE_SECRET_KEY"]
        self.LANGFUSE_PUBLIC_KEY = self.env["LANGFUSE_PUBLIC_KEY"]
        self.LANGFUSE_BASE_URL = self.env["LANGFUSE_BASE_URL"]

        # --- YAML CONFIGURATION ---
        self.models = self._load_yaml("models.yaml")
        self.params = self._load_yaml("param.yaml")

        # Convenient shortcuts for common sections.
        self.model_chat = self.models.get("openai", {}).get("chat", {})
        self.model_embedding = self.models.get("openai", {}).get("embedding", {})

        self.llm = self.params.get("llm", {})
        self.embedding = self.params.get("embedding", {})
        self.chunking = self.params.get("chunking", {})
        self.retrieval = self.params.get("retrieval", {})
        self.cag = self.params.get("cag", {})
        self.crag = self.params.get("crag", {})
        self.paths = self.params.get("paths", {})
        self.logging = self.params.get("logging", {})
        self.observability = self.params.get("observability", {})
        self.qdrant = self.params.get("qdrant", {})

    def _load_yaml(self, file_name: str) -> Dict[str, Any]:
        file_path = os.path.join(self.config_dir, file_name)
        if not os.path.exists(file_path):
            return {}

        with open(file_path, "r", encoding="utf-8") as file:
            content = yaml.safe_load(file)

        return content if isinstance(content, dict) else {}


settings = Config()
