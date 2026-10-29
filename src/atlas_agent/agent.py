"""Milestone 2: complete the bounded decision–action–observation loop."""

# Ruff suppressions cover names intentionally reserved for the TODOs. Remove
# this line after completing Milestone 2.
# ruff: noqa: F401, F821, F841

from __future__ import annotations

import json

from pydantic import ValidationError

from .agent_models import (
    ClarificationDecision,
    DecisionAdapter,
    FinalDecision,
    ToolCallDecision,
)
from .llm import StructuredLLM
from .models import AgentResponse, AgentRun, LabRequest, ToolObservation, TraceEvent
from .prompts import initialize_messages
from .tools import MUTATING_TOOLS, ToolEnvironment
from .validation import validate_final_response, validate_proposed_tool_call


def _failed(message: str, trace: list[TraceEvent], reason: str) -> AgentRun:
    return AgentRun(
        response=AgentResponse(status="failed", message=message),
        trace=trace,
        termination_reason=reason,
    )


def run_agent(
    request: LabRequest,
    *,
    llm: StructuredLLM,
    environment: ToolEnvironment,
    max_steps: int = 10,
) -> AgentRun:
    messages = initialize_messages(request)
    trace: list[TraceEvent] = []

    for step in range(1, max_steps + 1):
        try:
            decision = llm.generate(messages=messages, response_model=DecisionAdapter)
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            return _failed(
                f"The model did not produce a valid decision: {exc}",
                trace,
                "invalid_model_decision",
            )

        if isinstance(decision, ClarificationDecision):
            return AgentRun(
                response=AgentResponse(
                    status="clarification_required", message=decision.question
                ),
                trace=trace,
                termination_reason="clarification_required",
            )

        if isinstance(decision, FinalDecision):
            validation_result = validate_final_response(
                response=decision.response, trace=trace
            )

            if len(validation_result) > 0:
                return _failed(
                    message="; ".join(validation_result),
                    trace=trace,
                    reason="invalid final decision",
                )

            return AgentRun(
                response=decision.response,
                trace=trace,
                termination_reason="model_finished",
            )

        if not isinstance(decision, ToolCallDecision):
            return _failed(
                "Unsupported decision type.", trace, "invalid_model_decision"
            )

        tool_call_result = validate_proposed_tool_call(
            decision=decision, request=request, trace=trace
        )
        if len(tool_call_result) > 0:
            observation = ToolObservation(
                ok=False,
                tool=decision.tool,
                error_code="action_gate_denied",
                message="; ".join(tool_call_result),
            )
        else:
            observation = environment.dispatch(
                tool=decision.tool, arguments=decision.arguments
            )

        can_mutate = None
        if decision.tool in MUTATING_TOOLS:
            can_mutate = True
        else:
            can_mutate = False
        trace.append(
            TraceEvent(
                step=step,
                decision=decision.model_dump(),
                observation=observation,
                mutating=can_mutate,
            )
        )

        messages.append(
            {"role": "assistant", "content": json.dumps(decision.model_dump())}
        )
        messages.append(
            {"role": "tool", "content": json.dumps(observation.model_dump())}
        )

    return _failed(
        message="The agent exceeded the maximum number of steps.",
        trace=trace,
        reason="maximum_steps_exceeded",
    )
