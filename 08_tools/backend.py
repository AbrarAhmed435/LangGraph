from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Literal,Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langsmith import traceable

#############            TOOLs               ########################
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


search_tool=DuckDuckGoSearchRun()

# config={
#     "configurable":{"thread_id":"1"}
# }

@tool("calculator",description="Tool for calculator")
def calculator(first_num:float,second_num:float,operation:str)->dict:
    try:
        if operation=="add":
            result=first_num+second_num
        elif operation=="sub":
            result=first_num-second_num
        elif operation=="mul":
            result=first_num*second_num
        elif operation=="div":
            if not second_num:
                return {
                    "error":"Division by zero is not allowed"
                }
            result=first_num/second_num
        return {
            "result":result
        }
    except Exception as e:
        return {
            "error":str(e)
        }
            
import requests
import os

@tool("get_stock_prices",description="Tool to get live stock info")
def stock_prices(symbol:str)->dict:
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={os.environ['STOCK_API_KEY']}'
    r = requests.get(url)
    data = r.json()
    return data

API_KEY = os.getenv("WEATHER_API_KEY")  # keep key in .env
BASE_URL = "http://api.weatherapi.com/v1"

@tool("get_current_weather",description="Know current weather of give location")
def get_current_weather(city):
    url = f"{BASE_URL}/current.json"
    params = {
        "key": API_KEY,
        "q": city
    }
    response = requests.get(url, params=params)
    return response.json()


####################################################################

import os
os.environ['LANGCHAIN_PROJECT']="chatbot"

llm=HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="zai-org/GLM-4.7-Flash",
    repo_id="openai/gpt-oss-20b",
    # repo_id="openai/gpt-oss-120b",
    # repo_id="HuggingFaceH4/zephyr-7b-gemma-v0.1",
    # repo_id="lmsys/vicuna-13b-v1.5",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)
# generator=ChatHuggingFace(llm=llm)

tools=[stock_prices,calculator,get_current_weather]
model=model.bind_tools(tools)
tool_node=ToolNode(tools)


from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]
    
# model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
# model1=ChatOpenAI(model='gpt-4.1-mini')

@traceable(name="chat_node")
def chat_node(state:ChatState):
    messages=state['messages'][-10:]
    response=model.invoke(messages)

    return {
        'messages':[response]
    }

# print("apenai:", os.getenv("OPENAI_API_KEY"))
import os
print("CWD:", os.getcwd())

from dotenv import find_dotenv, load_dotenv

env_path = find_dotenv()
print("Loaded .env from:", env_path)

load_dotenv(env_path)

############SQLITE SETUP ###########################
connection=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=connection)

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)


graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge("tools","chat_node")

workflow=graph.compile(checkpointer=checkpointer)
# workflow=graph.compile()

@traceable(name="retrieve_all_threads")
def retrieve_all_threads():
    all_thread=set()
    for checkpoint in checkpointer.list(None): #returns all checkpoints 
        all_thread.add(checkpoint.config['configurable']['thread_id'])
    return all_thread

# print(workflow.invoke({
#     "messages":[HumanMessage(content="What is current weather in auckland")]
# }))

# workflow.invoke(
#         {"messages": [HumanMessage(content="What is my name")]},
#         config=config,
#         stream_mode="messages",  
#         )

# for message_chunk, metadata in workflow.stream(
#     {"messages": [HumanMessage(content="What is capital of india, and why give detailed report?")]},
#     config=config,
#     stream_mode="messages",  
# ):
#      if message_chunk.content:
#         print(message_chunk.content, end="|", flush=True)

# conversation=[]
# thread_id='2'
# while True:
#     user_message=input("Type here ....")
#     print(f"User: {user_message}")
#     if user_message.strip().lower() in ['exit','quit','bye']:
#         break
#     config={
#         "configurable":{'thread_id':thread_id}
#     }
#     # conversation.append(HumanMessage(content=user_message))
#     response = workflow.invoke(
#     {'messages': [HumanMessage(content=user_message)]},
#     config=config
# )
#     # ai_message=response['messages'][-1]
#     print("Ai:",response['messages'][-1].content)
#     # conversation.append(ai_message)

# workflow.get_state(config=config)
# workflow.get_state(config=config).values['messages']