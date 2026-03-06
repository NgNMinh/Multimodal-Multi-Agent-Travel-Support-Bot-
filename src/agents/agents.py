import os
from datetime import date, datetime
from typing import Callable, List, Optional

from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

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
from src.utils.prompt import prompt

load_dotenv()

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

weather_tool = load_tools(["openweathermap-api"])
api_key = os.getenv("GEMINI_API_KEY")
model_id = "gemini-2.0-flash"

llm = ChatGoogleGenerativeAI(
    model=model_id,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=api_key,
)


# ============================================================================
# BASE CLASSES
# ============================================================================

class Assistant:
    """Base assistant class that wraps a runnable and handles empty responses."""
    
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)
            state = {**state, "user_id": user_id}
            result = self.runnable.invoke(state)
            
            # If the LLM returns an empty response, re-prompt for actual response
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """Tool to mark task as completed or escalate to main assistant."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(..., description="The formatted memory to be stored")


# ============================================================================
# DELEGATION MODELS (for Primary Assistant)
# ============================================================================

class ToFlightBookingAssistant(BaseModel):
    """Transfers work to flight booking assistant."""

    request: str = Field(
        description="Any necessary followup questions the booking flight assistant should clarify before proceeding."
    )


class ToHotelBookingAssistant(BaseModel):
    """Transfers work to hotel booking assistant."""

    request: str = Field(
        description="Any additional information or requests from the user regarding the hotel booking."
    )


class ToTourBookingAssistant(BaseModel):
    """Transfers work to tour booking assistant."""

    request: str = Field(
        description="The user's tour-related request that will be passed to the tour booking assistant."
    )


class ToBookAirportShuttle(BaseModel):
    """Transfers work to shuttle booking assistant."""

    from_airport: str = Field(description="The airport where the user wants to be picked up.")
    to: str = Field(description="The destination after leaving the airport.")
    pickup_datetime: datetime = Field(description="The exact date and time for the pickup.")
    request: str = Field(
        description="Any additional requests, such as child seat, extra luggage space, or specific car model."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "from_airport": "SGN",
                "to_destination": "Quận 1",
                "pickup_datetime": "2025-03-21T10:00:00",
                "special_request": "tôi cần đặt xe từ sgn đến quận 1 ngày 21/3/2025 lúc 10h."
            }
        }


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

# Common instructions for specialized assistants
SPECIALIZED_ASSISTANT_INSTRUCTIONS = (
    "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
    '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. '
    "Do not make up invalid tools or functions."
    "\n\nSome examples for which you should CompleteOrEscalate:\n"
    " - 'what's the weather like this time of year?'\n"
    " - 'nevermind i think I'll book separately'\n"
    " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
)

flight_booking_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for handling flight searches and bookings. "
        "The primary assistant delegates work to you whenever the user needs help with flights. "
        "CONFIRM the booked flight details with the customer and inform them of any additional fees. "
        "When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        "\n\nCurrent user flight information:\n<Flights>\n{user_id}\n</Flights>"
        + SPECIALIZED_ASSISTANT_INSTRUCTIONS +
        "## Recall Memories\n"
        "Recall memories are contextually retrieved based on the conversations:\n{recall_memories}\n\n",
    ),
    ("placeholder", "{messages}"),
])

book_hotel_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for handling hotel bookings. "
        "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
        "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
        "When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        + SPECIALIZED_ASSISTANT_INSTRUCTIONS +
        " - 'Hotel booking confirmed'\n"
        "## Recall Memories\n"
        "Recall memories are contextually retrieved based on the conversations:\n{recall_memories}\n\n",
    ),
    ("placeholder", "{messages}"),
])

tour_booking_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for recommending travel tours based on user preferences. "
        "The primary assistant delegates to you whenever the user is interested in finding or booking a tour. "
        "Search for available tours based on the user's preferences and CONFIRM the booking details with the customer. "
        "When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        + SPECIALIZED_ASSISTANT_INSTRUCTIONS +
        " - 'tour booking confirmed'\n"
        "## Recall Memories\n"
        "Recall memories are contextually retrieved based on the conversations:\n{recall_memories}\n\n",
    ),
    ("placeholder", "{messages}"),
])

book_shuttle_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant for handling shuttle bookings. "
        "The primary assistant delegates work to you whenever the user needs help booking a shuttle. "
        "Search for available shuttle based on the user's preferences and CONFIRM the booking details with the customer. "
        "When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
        "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        + SPECIALIZED_ASSISTANT_INSTRUCTIONS +
        " - 'Shuttle booking confirmed'\n"
        "## Recall Memories\n"
        "Recall memories are contextually retrieved based on the conversations:\n{recall_memories}\n\n",
    ),
    ("placeholder", "{messages}"),
])

primary_assistant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful customer support assistant. "
        "Your primary role is to advise and answer customer questions. "
        "If a customer requests to search or book a flight, a shuttle, a hotel, or get tour recommendations, "
        "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. "
        "You are not able to make these types of changes yourself. "
        "Only the specialized assistants are given permission to do this for the user. "
        "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
        "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
        "When searching, be persistent. Expand your query bounds if the first search returns no results. "
        "If a search comes up empty, expand your search before giving up."
        "\n\nCurrent user flight information:\n<Flights>\n{user_id}\n</Flights>"
        "## Recall Memories\n"
        "Recall memories are contextually retrieved based on the conversations:\n{recall_memories}\n\n"
    ),
    ("placeholder", "{messages}"),
])


# ============================================================================
# RUNNABLES (Agent + Tools)
# ============================================================================

# Flight booking
book_flight_tools = [search_flights, book_flight]
book_flight_runnable = flight_booking_prompt | llm.bind_tools(book_flight_tools + [CompleteOrEscalate])

# Hotel booking
book_hotel_tools = [search_hotels, book_hotel]
book_hotel_runnable = book_hotel_prompt | llm.bind_tools(book_hotel_tools + [CompleteOrEscalate])

# Tour booking
book_tour_tools = [lookup_available_tours]
book_tour_runnable = tour_booking_prompt | llm.bind_tools(book_tour_tools + [CompleteOrEscalate])

# Shuttle booking
book_shuttle_tools = [search_shuttles, book_shuttle]
book_shuttle_runnable = book_shuttle_prompt | llm.bind_tools(book_shuttle_tools + [CompleteOrEscalate])

# Primary assistant
assistant_runnable = primary_assistant_prompt | llm.bind_tools([
    ToFlightBookingAssistant,
    ToBookAirportShuttle,
    ToTourBookingAssistant,
    ToHotelBookingAssistant,
    get_popular_tourist_destinations
] + weather_tool)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    """Create an entry node for transitioning to a specialized assistant."""
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user. "
                    f"The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name}, "
                    "and the booking, search, or other action is not complete until after you have successfully invoked the appropriate tool. "
                    "If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control "
                    "and immediately proceed to assist the user without asking for confirmation. "
                    "Do not mention that the assistant has changed or transferred — just respond naturally and proceed with the task.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }
    return entry_node


# Memory extractor
extractor = prompt | llm.with_structured_output(MemoryAnalysis)
