from atlas_agent.models import LabRequest
from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.validation import validate_final_response

REQUEST = LabRequest(
    authenticated_user_id="alice",
    text="Reserve tensile-01 for polymer-study from 2026-10-15 10:00 to 11:00.",
    project_id="polymer-study",
    resource_id="tensile-01",
    start="2026-10-15T10:00:00",
    end="2026-10-15T11:00:00",
)


def tool(tool, arguments):
    return {
        "kind": "tool_call",
        "tool": tool,
        "arguments": arguments,
        "reasoning_summary": "Obtain required evidence before acting.",
    }


def test_unavailable_resource(environment):
    """This test ensures that if a resource is unavailable (due to an existing reservation
    at the same time), the tool call will not be executed."""
    interval = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}
    decisions = [
        tool(
            "search_lab_documents",
            {
                "query": "reservation policy",
                "top_k": 8,
            },
        ),
        tool("get_user_authorizations", {"user_id": "alice"}),
        tool("get_resource_status", {"resource_id": "tensile-01", **interval}),
        tool(
            "reserve_resource",
            {
                "user_id": "alice",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **interval,
            },
        ),
        {
            # deterministic
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "The resource is unavailable.",
                "citations": [],
                "completed_actions": [],
            },
        },
    ]

    # add another reservation during the same interval so that the resource becomes unavailable during 10-11
    environment.simulator.state["reservations"].append(
        {
            "reservation_id": "existing-reservation",
            "user_id": "bob",
            "project_id": "another-project",
            "resource_id": "tensile-01",
            "start": "2026-10-15T10:00:00",
            "end": "2026-10-15T11:00:00",
            "status": "confirmed",
        }
    )

    reservations_before = len(environment.simulator.state["reservations"])
    run = run_agent(
        request=REQUEST, llm=ScriptedLLM(decisions=decisions), environment=environment
    )
    reservations_after = len(environment.simulator.state["reservations"])

    denied = run.trace[-1]
    assert denied.observation.ok == False
    assert denied.observation.error_code == "action_gate_denied"
    assert denied.observation.message is not None
    assert "resource_unavailable" in denied.observation.message
    assert reservations_before == reservations_after


def test_contradictory_responses(environment):
    environment.simulator.reset()
    interval = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}
    decisions = [
        tool(
            "search_lab_documents",
            {
                "query": "reservation policy",
                "top_k": 8,
            },
        ),
        tool("get_user_authorizations", {"user_id": "alice"}),
        tool("get_resource_status", {"resource_id": "tensile-01", **interval}),
        tool(
            "reserve_resource",
            {
                "user_id": "alice",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **interval,
            },
        ),
        {
            # deterministic
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "The reservation was not completed.",
                "citations": [],
                "completed_actions": [],
            },
        },
    ]

    run = run_agent(
        request=REQUEST, llm=ScriptedLLM(decisions=decisions), environment=environment
    )

    successful_reservation = False

    for event in run.trace:
        if (
            event.observation.tool == "reserve_resource"
            and event.observation.ok == True
        ):
            successful_reservation = True
    assert successful_reservation

    assert run.response.status == "failed"
    assert "not_completed_with_successful_mutation" in run.response.message


def test_unapproved_after_hours_reservation(environment):
    environment.simulator.reset()
    interval = {"start": "2026-10-15T19:00:00", "end": "2026-10-15T20:00:00"}
    decisions = [
        tool(
            "search_lab_documents",
            {
                "query": "reservation after-hours approval policy",
                "top_k": 8,
            },
        ),
        tool("get_user_authorizations", {"user_id": "bob"}),
        tool("get_resource_status", {"resource_id": "tensile-01", **interval}),
        tool(
            "reserve_resource",
            {
                "user_id": "bob",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **interval,
            },
        ),
        {
            # deterministic
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "After-hours approval was not found.",
                "citations": [],
                "completed_actions": [],
            },
        },
    ]

    after_hours_request = LabRequest(
        authenticated_user_id="bob",
        text="Reserve tensile-01 for polymer-study from 7 PM to 8 PM.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T19:00:00",
        end="2026-10-15T20:00:00",
    )
    reservations_before = len(environment.simulator.state["reservations"])
    run = run_agent(
        request=after_hours_request,
        llm=ScriptedLLM(decisions=decisions),
        environment=environment,
    )
    reservations_after = len(environment.simulator.state["reservations"])

    denied = run.trace[-1]
    assert denied.observation.ok == False
    assert denied.observation.error_code == "action_gate_denied"
    assert denied.observation.message is not None
    assert "after_hours_approval_missing" in denied.observation.message
    assert reservations_before == reservations_after
