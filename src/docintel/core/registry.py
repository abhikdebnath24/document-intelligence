from __future__ import annotations

from collections.abc import Callable

from docintel.core.errors import UnknownStrategyError


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._factories:
                raise ValueError(f"{self.kind} {name!r} already registered")
            self._factories[name] = cls
            return cls

        return decorator

    def create(self, name: str, **params: object) -> T:
        if name not in self._factories:
            raise UnknownStrategyError(self.kind, name, self.names())
        return self._factories[name](**params)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def get(self, name: str) -> type[T]:
        if name not in self._factories:
            raise UnknownStrategyError(self.kind, name, self.names())
        return self._factories[name]
