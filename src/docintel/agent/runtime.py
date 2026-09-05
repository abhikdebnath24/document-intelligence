from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from docintel.agent.cache import ChunkCache
from docintel.config import AppConfig
from docintel.core.errors import DeadlineExceeded
from docintel.core.interfaces import BaseGrader, BaseGroundednessVerifier
from docintel.llm.factory import StructuredCaller
from docintel.llm.prompts import PromptBank


@dataclass
class AgentRuntime:
    config: AppConfig
    caller: StructuredCaller
    prompts: PromptBank
    cache: ChunkCache
    grader: BaseGrader
    verifier: BaseGroundednessVerifier
    pipeline: Any
    started: float
    query_id: str

    def check_deadline(self) -> None:
        budget = self.config.llm.query_deadline_s
        if time.monotonic() - self.started > budget:
            raise DeadlineExceeded(f"query_deadline_s={budget} exceeded")
