from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool, tool

from tools import toolset  # your other domain tools (e.g., search)

load_dotenv()

# --- LLM ---
# LLM used ONLY by the manager – hard-wired to call manager_decision
manager_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# Generic LLM for all sub-agents (no forced tool)
worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Memory ---
def make_memory(llm_obj):
    """Return a private summary buffer for whichever LLM this agent uses."""
    return ConversationSummaryBufferMemory(
        llm=llm_obj,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=32000,
    )

# Manager uses its own constrained LLM
manager_memory = make_memory(manager_llm)

# Workers use an unconstrained clone (worker_llm)
ds_memory = make_memory(worker_llm)
ba_memory = make_memory(worker_llm)



# --- Cached sub-agents (DS & BA) ---
_data_scientist_agent = AgentExecutor(
    agent=create_openai_functions_agent(
        llm=worker_llm,
        tools=toolset,
        prompt=ChatPromptTemplate.from_messages([
            ("system", "You are a data scientist. Answer data questions using your tools."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
    ),
    tools=toolset,
    memory=ds_memory,
    verbose=True
)

_business_analyst_agent = AgentExecutor(
    agent=create_openai_functions_agent(
        llm=worker_llm,
        tools=toolset,
        prompt=ChatPromptTemplate.from_messages([
            ("system", "You are a business analyst. Answer business questions using your tools."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
    ),
    tools=toolset,
    memory=ba_memory,
    verbose=True
)

# --- DS/BA Tools for manager to call ---
@tool
def data_scientist_tool(input: str) -> str:
    """Ask data-science-related questions like analysis, modeling, or trends."""
    return _data_scientist_agent.invoke({"input": input})["output"]

@tool
def business_analyst_tool(input: str) -> str:
    """Ask business-related questions like pricing, strategy, or market insights."""
    return _business_analyst_agent.invoke({"input": input})["output"]

subagent_tools = [data_scientist_tool, business_analyst_tool]


# --- Manager tool schema enforcement ---
class ManagerDecisionArgs(BaseModel):
    route: Literal["data_scientist", "business_analyst", "end"] = Field(
        description="Where to delegate next."
    )
    output: str = Field(description="Message to give back to the user.")

def manager_decision(route: str, output: str) -> dict:
    """Return the arguments exactly as provided by the model."""
    # Simply echo back in a schema-consistent dict
    return {"route": route, "output": output}


manager_tool = StructuredTool.from_function(
    func=manager_decision,
    name="manager_decision",
    description="Manager must call this and provide both `route` "
                "and a user-facing `output`.",
    args_schema=ManagerDecisionArgs,
    return_direct=False,          # what the function returns goes back to the agent
)

# --- Manager prompt ---
manager_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Manager agent responsible for solving complex business and data science problems.

        You have access to two expert agents:
        - `ask_business_analyst`: defines strategies, pricing models, and market approaches.
        - `ask_data_scientist`: implements code, runs analysis, builds models, and answers technical questions.

        You can solve problems on your own when confident.
        Otherwise, delegate part of the task by calling the appropriate tool.

        The user cannot see what the BA or DS are doing. You need to forward their responses to the user.
        Always return a final decision using the `manager_decision` function.
        """
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# --- Manager agent ---
manager_agent = AgentExecutor(
    agent=create_openai_functions_agent(
        llm=manager_llm,
        tools=[manager_tool, *subagent_tools],
        prompt=manager_prompt
    ),
    tools=[manager_tool, *subagent_tools],
    memory=manager_memory,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=5,
)

