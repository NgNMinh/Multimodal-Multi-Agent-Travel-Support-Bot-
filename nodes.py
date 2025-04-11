from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from state import State
from agents import Assistant, assistant_runnable, create_entry_node, book_flight_runnable, CompleteOrEscalate, book_shuttle_runnable, ToBookAirportShuttle, ToFlightBookingAssistant
from tools import search_flights, book_flight, search_shuttles, book_shuttle
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from typing import Annotated, Literal, Optional

graph_builder = StateGraph(State)

graph_builder.add_node("primary_assistant", Assistant(assistant_runnable))

graph_builder.add_node(
    "enter_book_flight",
    create_entry_node("Flight Searching & Booking Assistant", "book_flight"),
)
graph_builder.add_node("book_flight", Assistant(book_flight_runnable))
graph_builder.add_edge("enter_book_flight", "book_flight")
graph_builder.add_node(
    "book_flight_tools",
    ToolNode([search_flights, book_flight]),
)


def route_book_flight(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "book_flight_tools"


graph_builder.add_edge("book_flight_tools", "book_flight")
graph_builder.add_conditional_edges(
    "book_flight",
    route_book_flight,
    ["book_flight_tools", "leave_skill", END],
)

# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


graph_builder.add_node("leave_skill", pop_dialog_state)
graph_builder.add_edge("leave_skill", "primary_assistant")

graph_builder.add_node(
    "enter_book_shuttle",
    create_entry_node("Shuttle Assistant", "book_shuttle"),
)
graph_builder.add_node("book_shuttle", Assistant(book_shuttle_runnable))
graph_builder.add_edge("enter_book_shuttle", "book_shuttle")
graph_builder.add_node(
    "book_shuttle_tools",
    ToolNode([search_shuttles, book_shuttle]),
)
def route_book_shuttle(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "book_shuttle_tools"


graph_builder.add_edge("book_shuttle_tools", "book_shuttle")
graph_builder.add_conditional_edges(
    "book_shuttle",
    route_book_shuttle,
    [
        "book_shuttle_tools",
        "leave_skill",
        END,
    ],
)

def route_primary_assistant(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_book_flight"
        elif tool_calls[0]["name"] == ToBookAirportShuttle.__name__:
            return "enter_book_shuttle"
    raise ValueError("Invalid route")

graph_builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_book_flight",
        "enter_book_shuttle",
        END,
    ],
)

# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "book_flight",
    "book_shuttle",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


graph_builder.add_conditional_edges(START, route_to_workflow)

#graph_builder.set_entry_point("primary_assistant")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# graph_image = graph.get_graph(xray=True).draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(graph_image)  # Chỉ cần ghi trực tiếp bytes vào file
#
# print("Graph saved as graph.png, open it manually.")
