import uuid
from typing import Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.agents import (
    Assistant,
    CompleteOrEscalate,
    ToBookAirportShuttle,
    ToFlightBookingAssistant,
    ToHotelBookingAssistant,
    ToTourBookingAssistant,
    assistant_runnable,
    book_flight_runnable,
    book_hotel_runnable,
    book_shuttle_runnable,
    book_tour_runnable,
    create_entry_node,
    extractor,
    weather_tool,
)
from src.core.state import State
from src.database.db import vector_store
from src.tools.tools import (
    book_flight,
    book_hotel,
    book_shuttle,
    get_popular_tourist_destinations,
    lookup_available_tours,
    search_flights,
    search_hotels,
    search_recall_memories,
    search_shuttles,
)


# ============================================================================
# MEMORY FUNCTIONS
# ============================================================================

def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    recall_memories = search_recall_memories.invoke(state["messages"][-1].content, config)
    return {
        "recall_memories": recall_memories,
    }


def extract_memories(state: State, config: RunnableConfig):
    """Process the current state and generate a response using the LLM.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        dict: Empty dict after processing memories.
    """
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    
    analysis = extractor.invoke(
        {
            "messages": [HumanMessage(state['messages'][-1].content)],
            "recall_memories": recall_str,
        }
    )
    
    if analysis.is_important and analysis.formatted_memory:
        configuration = config.get("configurable", {})
        user_id = configuration.get("user_id", None)
        
        document = Document(
            page_content=analysis.formatted_memory,
            id=str(uuid.uuid4()),
            metadata={"user_id": user_id}
        )
        vector_store.add_documents([document])
    return {}


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_book_flight(state: State):
    """Route flight booking based on tool calls."""
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "book_flight_tools"


def route_book_hotel(state: State):
    """Route hotel booking based on tool calls."""
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "book_hotel_tools"


def route_book_tour(state: State):
    """Route tour booking based on tool calls."""
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "book_tour_tools"


def route_book_shuttle(state: State):
    """Route shuttle booking based on tool calls."""
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "book_shuttle_tools"


def route_primary_assistant(state: State):
    """Route from primary assistant to specialized assistants."""
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_book_flight"
        elif tool_calls[0]["name"] == ToBookAirportShuttle.__name__:
            return "enter_book_shuttle"
        elif tool_calls[0]["name"] == ToTourBookingAssistant.__name__:
            return "enter_book_tour"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


def route_to_workflow(
    state: State,
) -> Literal["primary_assistant", "book_flight", "book_shuttle", "book_tour", "book_hotel"]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant."""
    messages = []
    if state["messages"][-1].tool_calls:
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


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

graph_builder = StateGraph(State)

# Memory nodes
graph_builder.add_node("load_memories", load_memories)
graph_builder.add_node("extract_memories", extract_memories)
graph_builder.add_edge(START, "load_memories")
graph_builder.add_edge("load_memories", "extract_memories")

# Primary assistant
graph_builder.add_node("primary_assistant", Assistant(assistant_runnable))
graph_builder.add_node("primary_assistant_tools", ToolNode([get_popular_tourist_destinations] + weather_tool))
graph_builder.add_edge("primary_assistant_tools", "primary_assistant")

# Shared exit node
graph_builder.add_node("leave_skill", pop_dialog_state)
graph_builder.add_edge("leave_skill", "primary_assistant")

# Flight booking assistant
graph_builder.add_node("enter_book_flight", create_entry_node("Flight Searching & Booking Assistant", "book_flight"))
graph_builder.add_node("book_flight", Assistant(book_flight_runnable))
graph_builder.add_node("book_flight_tools", ToolNode([search_flights, book_flight]))
graph_builder.add_edge("enter_book_flight", "book_flight")
graph_builder.add_edge("book_flight_tools", "book_flight")
graph_builder.add_conditional_edges("book_flight", route_book_flight, ["book_flight_tools", "leave_skill", END])

# Hotel booking assistant
graph_builder.add_node("enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel"))
graph_builder.add_node("book_hotel", Assistant(book_hotel_runnable))
graph_builder.add_node("book_hotel_tools", ToolNode([search_hotels, book_hotel]))
graph_builder.add_edge("enter_book_hotel", "book_hotel")
graph_builder.add_edge("book_hotel_tools", "book_hotel")
graph_builder.add_conditional_edges("book_hotel", route_book_hotel, ["leave_skill", "book_hotel_tools", END])

# Tour booking assistant
graph_builder.add_node("enter_book_tour", create_entry_node("Tour Searching Assistant", "book_tour"))
graph_builder.add_node("book_tour", Assistant(book_tour_runnable))
graph_builder.add_node("book_tour_tools", ToolNode([lookup_available_tours]))
graph_builder.add_edge("enter_book_tour", "book_tour")
graph_builder.add_edge("book_tour_tools", "book_tour")
graph_builder.add_conditional_edges("book_tour", route_book_tour, ["book_tour_tools", "leave_skill", END])

# Shuttle booking assistant
graph_builder.add_node("enter_book_shuttle", create_entry_node("Shuttle Assistant", "book_shuttle"))
graph_builder.add_node("book_shuttle", Assistant(book_shuttle_runnable))
graph_builder.add_node("book_shuttle_tools", ToolNode([search_shuttles, book_shuttle]))
graph_builder.add_edge("enter_book_shuttle", "book_shuttle")
graph_builder.add_edge("book_shuttle_tools", "book_shuttle")
graph_builder.add_conditional_edges("book_shuttle", route_book_shuttle, ["book_shuttle_tools", "leave_skill", END])

# Primary assistant routing
graph_builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    ["enter_book_flight", "enter_book_shuttle", "primary_assistant_tools", "enter_book_tour", "enter_book_hotel", END],
)

# Route from memory extraction to appropriate workflow
graph_builder.add_conditional_edges("extract_memories", route_to_workflow)

# Compile graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
