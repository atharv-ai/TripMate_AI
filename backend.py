import os
from dotenv import load_dotenv

load_dotenv()


from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import(
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE url is missing"
        )

    if "sslmode=" not in database_url:
        seperator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{seperator}sslmode=require"

    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ api key is missing"
    )

# llm

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)

# state

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# flight agent

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return{
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="FLight results fetched.")
        ],
        "llm_calls": state.get("llm_calls",0) + 1
    }

# hotet agent

def hotel_agent(state: TravelState):
    query = f"best hotel for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel info fetched.")
        ],
        "llm_calls": state.get("llm_calls",0) + 1
    }

# itinerary agent

def itinerary_agent(state: TravelState):
    
    prompt = f"""
Create a practical day-by-day travel itinerary.

User Query:
{state["user_query"]}

Flight Results:
{state["flight_results"]}

Hotel Results:
{state["hotel_results"]}

Instructions:
- Create a separate section for each day.
- Include 3-5 important activities per day.
- Mention useful places and activities.
- Keep the schedule practical and budget-aware.
- Do not repeat flight or hotel information.
- Keep each day's description concise.
- Do not invent places or details.
- Return only the itinerary in Markdown.
"""


    response = llm.invoke([
        SystemMessage(content="You are an exprest travel planner."),
        HumanMessage(content = prompt)
    ])

    return{
        "itinerary": response.content,
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# final response

def final_response_agent(state: TravelState):
    
    final_prompt = f"""
    You are a professional travel assistant. Generate a clear, practical, and well-structured final travel plan using the information provided below.

    USER REQUEST:
    {state["user_query"]}

    FLIGHT RESULTS:
    {state["flight_results"]}

    HOTEL RESULTS:
    {state["hotel_results"]}

    ITINERARY:
    {state["itinerary"]}

    Follow this exact response structure:

    # ✈️ Trip Summary
    - Briefly summarize the destination, duration, and purpose of the trip.
    - Mention important details from the user's request.

    # 1. Flight Information
    - Present the available flight options using bullet points.
    - Include airline, departure, arrival, duration, and price wherever available.
    - If ticket prices are unavailable, clearly mention that the live flight API may not provide ticket pricing.
    - Do not invent flight details or prices.

    # 2. Hotel Suggestions
    - Present available hotels using bullet points.
    - Include hotel name, location, price, rating, and other useful details wherever available.
    - Do not invent missing information.

    # 3. Day-by-Day Itinerary
    - Organize the itinerary under separate headings:
    ## Day 1
    ## Day 2
    And so on.
    - Include activities, places to visit, and useful details for each day.
    - Keep the schedule practical and avoid unnecessary repetition.

    # 4. Estimated Budget
    - Present the estimated budget in a Markdown table.
    - Include available cost categories such as flights, accommodation, food, transport, and activities.
    - Show the total estimated cost if sufficient pricing information is available.
    - Clearly label estimates and unavailable costs.
    - Do not invent prices or calculate totals using missing values.

    # 5. Final Recommendation
    - Give a short, practical recommendation based on the available flight, hotel, itinerary, and budget information.
    - Mention any important missing information the user should verify before booking.

    IMPORTANT RULES:
    - Use proper Markdown headings, bullet points, and tables.
    - Keep the response clear, readable, and useful for real travel planning.
    - Use only the information provided in the inputs.
    - Never fabricate flight details, hotel details, prices, ratings, or activities.
    - If any section has no available information, clearly say that the information is unavailable instead of making it up.
    - Avoid repeating the same information across sections.
    - Do not return the entire response as one paragraph.

    - Do not create or assume flight schedules, hotel details,
    prices, ratings, or activity costs.
    - Do not treat estimated prices as confirmed prices.
    - If flight options contain duplicate information,
    mention that the results appear duplicated.
    - Ensure hotel suggestions match the cities in the itinerary.
    - Check that the itinerary matches the requested trip duration.
    - Do not include destinations that are not part of the user's request.
    - Clearly identify missing or inconsistent information.
    """

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return{
        "messages": [response],
        "llm_calls": state.get("llm_calls",0) + 1
    }

# Build Graph

graph = StateGraph(TravelState)

graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("itinerary_agent",itinerary_agent)
graph.add_node("final_agent",final_response_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent",END)


# postgreSQL checkpoint

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

# function for fast api

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable":{
            "thread_id": thread_id
        }
    }
    result = travel_graph.invoke({
        "messages": [
            HumanMessage(content=user_input)
        ],
        "user_query": user_input,
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "llm_calls": 0
    },
    config=config
    )
    final_answer = result["messages"][-1].content

    return{
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results",""),
        "hotel_results": result.get("hotel_results",""),
        "itinerary": result.get("itinerary",""),
        "llm_calls": result.get("llm_calls",0)
    }