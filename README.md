Simple multi-agentic system. One manager, one business analyst and one data scientist.

Run with:
- chainlit: chainlit run chainlit_app.py
- streamlit: streamlit run streamlit_app.py

The BA and the DS have access to internet search.
The manager enforces routing with Pydantic. The flow is flexible, the manager can keep calling the BA and DS in any order until satisfied with the response. The system has a simple chatbot UI.
