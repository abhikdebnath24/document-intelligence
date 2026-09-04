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
