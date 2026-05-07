from langgraph.graph import StateGraph, START , END
from langgraph.graph.message import add_messages
from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
import asyncio
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    # repo_id="zai-org/GLM-4.7-Flash",
    repo_id="openai/gpt-oss-20b",
    # repo_id="openai/gpt-oss-120b",
    # repo_id="zlyngkhoi/qwen2_lrp_lora_3b",
    # repo_id="lmsys/vicuna-13b-v1.5",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)
# generator=ChatHuggingFace(llm=llm)

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

client=MultiServerMCPClient(
    {
        "math":{
            "transport":"stdio",
            "command":"python3",
            "args":["/home/abrar/Desktop/Abrar/LangGraph/09_MCP/main.py"]
        },
        "expense_tracker":{
            "transport":"sse",
            "url":"http://localhost:6277/sse",
            "headers":{
                "Authorization":"Bearer 9253e95f0c22ee66c3aac8434dd1350e0193408a96742eeffbc8a4dc4cb838dc"
            }
        }
    }
)
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


async def build_graph():

    tools=await client.get_tools()

    print(tools)
    bound_model=model.bind_tools(tools)

    async def chat_node(state:ChatState):
        messages=state['messages']
        response=await bound_model.ainvoke(messages)

        return {
            'messages':[response]
        }
    
    # tools=[tools]
    
    tool_node=ToolNode(tools)

    graph=StateGraph(ChatState)

    graph.add_node('chat_node',chat_node)
    graph.add_node('tools',tool_node)

    graph.add_edge(START,'chat_node')
    graph.add_conditional_edges('chat_node',tools_condition)
    graph.add_edge('tools',"chat_node")

    chatbot=graph.compile()

    return chatbot
async def main():
    chatbot=await build_graph()

    result=await chatbot.ainvoke({"messages":[HumanMessage(content="Find the multiplication of 7.99 and 5.55")]})
    
    # ainvoke -=> Asynchronous invoke
    
    print(result['messages'][-1].content)

    
if __name__=="__main__":
    asyncio.run(main())