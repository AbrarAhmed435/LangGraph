import streamlit as st
from sqlite_backend import workflow,retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid



###########UTILITY FUNCTIONS ######################################

def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id




def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]


if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]


def load_conversation(thread_id,message_history):
    state=workflow.get_state(config={'configurable':{'thread_id':thread_id}})
    if not state.values:
        return []
    return state.values['messages']

###############################################################3

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()
    


if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=list(retrieve_all_threads())

add_thread(st.session_state['thread_id'])


    
config={
    "configurable":{"thread_id":st.session_state['thread_id']},
    "metadata":{
        "thread_id":st.session_state["thread_id"]
    },
    "run_name":"chat_turn"
}

######################## SIDEBAR #####################################################

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header("My Conversations")

for chat_thread in reversed(st.session_state["chat_threads"]):
    is_active = chat_thread == st.session_state["thread_id"]

    label = f"🟢 {chat_thread}" if is_active else f"{chat_thread}"

    if st.sidebar.button(label, key=f"thread-{chat_thread}"):
        st.session_state["thread_id"] = chat_thread
        messages = load_conversation(chat_thread, st.session_state["message_history"])

        temp_message = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_message.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_message
        st.rerun()

#####################################################################################

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input=st.chat_input("Type Here")

if user_input:
    st.session_state['message_history'].append({'role':"user","content":user_input})
    buffer=""
    with st.chat_message('user'):
        st.text(user_input)

    for chunk, _ in workflow.stream(
        {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="chunk", 
    ):
        if chunk.content:
            buffer+=chunk.content
    st.session_state['message_history'].append({'role':'assistant','content':buffer})
    st.rerun()

