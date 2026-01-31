"""
Read docs : https://docs.langchain.com/oss/python/langgraph/streaming
"""




from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Literal,Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm=HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="HuggingFaceH4/zephyr-7b-gemma-v0.1",
    # repo_id="lmsys/vicuna-13b-v1.5",
    task="text-generation"
)
# model=ChatHuggingFace(llm=llm)
# generator=ChatHuggingFace(llm=llm)

from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]
    
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
# model1=ChatOpenAI(model='gpt-4.1-mini')
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
config={
    "configurable":{"thread_id":"1"}
}

checkpointer=MemorySaver()

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

workflow=graph.compile(checkpointer=checkpointer)

# The "messages" stream mode returns an iterator of tuples (message_chunk, metadata)
# where message_chunk is the token streamed by the LLM and metadata is a dictionary
# with information about the graph node where the LLM was called and other information

for message_chunk, metadata in workflow.stream(
    {"messages": [HumanMessage(content="What is capital of india, and why give detailed report?")]},
    config=config,
    stream_mode="messages",  
):
     if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)

