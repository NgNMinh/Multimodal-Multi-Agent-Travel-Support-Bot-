import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tools import search_flights, book_flight, search_shuttles, book_shuttle
from langchain_core.runnables import Runnable, RunnableConfig
from state import State
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from datetime import date,datetime
from typing import Callable

from langchain_core.messages import ToolMessage
load_dotenv()

# Get API key and define model id
api_key = os.getenv("GEMINI_API_KEY")
model_id = "gemini-2.0-flash"

# Create LLM class
llm = ChatGoogleGenerativeAI(
    model=model_id,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=api_key,
)


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)
            state = {**state, "user_id": user_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
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
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

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

flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling flight searches and books. "
            " The primary assistant delegates work to you whenever the user needs help updating their bookings. "
            "CONFIRM the booked flight details with the customer and inform them of any additional fees. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user flight information:\n<Flights>\n{user_id}\n</Flights>"
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',
        ),
        ("placeholder", "{messages}"),
    ]
)

book_flight_tools = [search_flights, book_flight]
book_flight_runnable = flight_booking_prompt | llm.bind_tools(
    book_flight_tools + [CompleteOrEscalate]
)


book_shuttle_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling shuttle bookings. "
            "The primary assistant delegates work to you whenever the user needs help booking a shuttle. "
            "Search for available shuttle based on the user's preferences and CONFIRM the booking details with the customer. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
            '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'what's the weather like this time of year?'\n"
            " - 'What flights are available?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Shuttle booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
)

book_shuttle_tools = [search_shuttles, book_shuttle]
book_shuttle_runnable = book_shuttle_prompt | llm.bind_tools(
    book_shuttle_tools + [CompleteOrEscalate]
)

# Primary Assistant
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight searching and booking."""

    request: str = Field(
        description="Any necessary followup questions the booking flight assistant should clarify before proceeding."
    )

class ToBookAirportShuttle(BaseModel):
    """Handles airport shuttle booking requests."""

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


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant."
            "Your primary role is to advise and answer customer questions. "
            "If a customer requests to search or book a flight, search or book a shuttle, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_id}\n</Flights>"
        ),
        ("placeholder", "{messages}"),
    ]
)

assistant_runnable = primary_assistant_prompt | llm.bind_tools([ToFlightBookingAssistant, ToBookAirportShuttle])

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, search, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node