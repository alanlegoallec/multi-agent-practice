import chainlit as cl
from graph import build_graph

graph = build_graph()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])

@cl.on_message
async def on_message(msg: cl.Message):
    history = cl.user_session.get("chat_history")

    # Save user message
    history.append(("user", msg.content))

    # Show a "thinking" placeholder
    thinking_msg = cl.Message(content="🧠 Thinking...")
    await thinking_msg.send()

    # Call LangGraph
    result = await graph.ainvoke({"input": msg.content})

    # Extract and set assistant reply
    output = result.get("output", "🤖 No output.")
    history.append(("assistant", output))
    thinking_msg.content = output  # Set new content
    await thinking_msg.update()    # Trigger the update

    # Show intermediate steps (if any)
    for step in result.get("intermediate_steps", []):
        role = step.get("role", "assistant")
        response = step.get("response", "")
        history.append((role, response))
        await cl.Message(author=role, content=response).send()

    cl.user_session.set("chat_history", history)
