from __future__ import annotations

import pytest

from docintel.core.device import resolve_device
from docintel.core.errors import UnknownStrategyError
from docintel.core.registry import Registry


class Greeter:
    def __init__(self, prefix: str = "hi") -> None:
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix} {name}"


def test_register_create_and_names() -> None:
    registry: Registry[Greeter] = Registry("greeter")

    @registry.register("basic")
    class Basic(Greeter):
        pass

    assert registry.names() == ["basic"]
    instance = registry.create("basic", prefix="yo")
    assert instance.greet("ada") == "yo ada"


def test_unknown_name_lists_known() -> None:
    registry: Registry[Greeter] = Registry("greeter")

    @registry.register("basic")
    class Basic(Greeter):
        pass

    with pytest.raises(UnknownStrategyError, match="unknown greeter 'missing'") as exc:
        registry.create("missing")
    assert exc.value.available == ["basic"]


def test_duplicate_register_rejected() -> None:
    registry: Registry[Greeter] = Registry("greeter")

    @registry.register("basic")
    class First(Greeter):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registry.register("basic")
        class Second(Greeter):
            pass


def test_resolve_device_explicit_and_invalid() -> None:
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu")
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
