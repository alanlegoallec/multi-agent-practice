import streamlit as st
from graph import build_graph

st.set_page_config(page_title="Multi-Agent Assistant", layout="centered")
st.title("🤖 Talk to Your Manager Agent")

graph = build_graph()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render chat history
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)

# User input
user_input = st.chat_input("Send a message to the Manager")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    result = graph.invoke({"input": user_input})

    # Render manager's summary
    manager_response = result.get("output", "No output.")
    st.session_state.chat_history.append(("assistant", manager_response))
    with st.chat_message("assistant"):
        st.markdown(manager_response, unsafe_allow_html=True)

    # Render any intermediate steps (e.g., sub-agent outputs)
    for step in result.get("intermediate_steps", []):
        role = step.get("role", "assistant")
        response = step.get("response", "")
        st.session_state.chat_history.append((role, response))
        with st.chat_message(role):
            st.markdown(response, unsafe_allow_html=True)
