# graph.py

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from agents import manager_agent


class AgentState(TypedDict):
    input: str
    output: str
    route: Literal["data_scientist", "business_analyst", "end"]
    chat_history: list
    intermediate_steps: list


def run_manager(state: AgentState) -> AgentState:
    """Invoke the manager agent with tool_choice routing enforced. Update memory manually."""
    result = manager_agent.invoke(
        {
            "input": state["input"],
            "chat_history": state.get("chat_history", []),
        },
        config={
            "tool_choice": {
                "type": "function",
                "function": {"name": "manager_decision"}
            }
        }
    )

    print("🧠 Manager agent returned:", result)

    # Append to intermediate steps if routing to a subagent
    updated_steps = state.get("intermediate_steps", [])
    if result.get("route") in {"data_scientist", "business_analyst"}:
        updated_steps = updated_steps + [{
            "role": result["route"],
            "response": result["output"]
        }]

    return {
        **state,
        "output": result["output"],
        "route": result.get("route", "end"),
        "chat_history": result.get("chat_history", []),
        "intermediate_steps": updated_steps
    }


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("manager", run_manager)

    builder.set_entry_point("manager")

    builder.add_conditional_edges(
        "manager",
        lambda state: state["route"],
        {
            "data_scientist": "manager",  # manager loops to itself
            "business_analyst": "manager",
            "end": END
        }
    )

    return builder.compile()
