class DocIntelError(Exception):
    """Base error for the package."""


class UnknownStrategyError(DocIntelError):
    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self.kind = kind
        self.name = name
        self.available = available
        super().__init__(f"unknown {kind} {name!r}; known: {', '.join(available) or '(none)'}")


class ConfigError(DocIntelError):
    """Invalid YAML, merge, or env override."""


class MissingSecretError(DocIntelError):
    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(f"required secret {env_var} is not set in the environment or .env")


class CollectionMismatchError(DocIntelError):
    """Existing Qdrant collection does not match this index_sig / vector schema."""


class QdrantInUseError(DocIntelError):
    """Embedded Qdrant path is locked by another opener in this process or another."""


class DeadlineExceeded(DocIntelError):
    """Whole-graph query_deadline_s elapsed."""


class StructuredOutputError(DocIntelError):
    """Chat model returned text that could not be parsed as the requested schema."""
