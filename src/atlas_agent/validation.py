"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    errors = []
    if "user_id" in decision.arguments:
        if decision.arguments.get("user_id") != request.authenticated_user_id:
            errors.append("user_id_must_match_authenticated_user")

    for field in {"project_id", "resource_id", "start", "end"}:
        if field not in decision.arguments:
            continue
        if getattr(request, field) is None:
            errors.append(f"{field}_unresolved_in_request")
        elif decision.arguments.get(field) != getattr(request, field):
            errors.append(f"{field}_does_not_match_request")
    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    errors = []
    error_codes = {
        "arguments_complete": "required_arguments_missing",
        "policy_retrieved": "required_policy_missing",
        "authorizations_checked": "authorizations_not_checked",
        "resource_status_checked": "resource_status_not_checked",
        "project_authorized": "project_not_authorized",
        "qualification_current": "required_qualification_missing",
        "resource_available": "resource_unavailable",
        "calibration_current": "calibration_not_current",
        "maintenance_clear": "maintenance_not_clear",
        "after_hours_approval_satisfied": "after_hours_approval_missing",
    }

    for field, error_code in error_codes.items():
        if not getattr(context, field):
            errors.append(error_code)
    return errors


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    errors = []

    retrieved_passages = retrieved_passage_ids(trace=trace)
    for citation in response.citations:
        if citation not in retrieved_passages:
            errors.append(f"citations_not_retrieved")

    completed_actions = response.completed_actions
    successful_mutations = []
    for event in trace:
        if event.mutating == True and event.observation.ok == True:
            # then the tool call made a successful mutation
            successful_mutations.append(event.observation.tool)
    for action in response.completed_actions:
        if action not in successful_mutations:
            errors.append(f"completed_action_not_in_successful_trace")

    if response.status == "completed":
        if len(successful_mutations) == 0:
            errors.append("completed_without_successful_mutation")
        if len(completed_actions) == 0:
            errors.append("completed_without_completed_actions")
    else:
        if len(successful_mutations) != 0:
            errors.append("not_completed_with_successful_mutation")
        if len(completed_actions) != 0:
            errors.append("not_completed_with_completed_actions")
    return errors
