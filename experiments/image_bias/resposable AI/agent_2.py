#!pip install protobuf==5.28.1 google-adk==1.0.0 litellm -q -q
#pip install Deprecated

from google.adk.tools import FunctionTool
from google.genai import types
from google import genai
from google.adk.models.lite_llm import LiteLlm
import litellm

AGENT_MODEL = LiteLlm(model="openai/gpt-4o-mini")
APP_NAME = "adk_course_app"
USER_ID = "user_123"
SESSION_ID = "support_session"

# Define a simple FAQ knowledge base
FAQ_DATA = {
    "return policy": "You can return items within 30 days of purchase.",
    "hours": "Our support team is available from 9am to 5pm, Monday to Friday.",
    "contact": "You can reach support at help@example.com."
}

# Define the tool function
def lookup_faq(question: str) -> str:
    faq_text = "\n".join(f"- {k}: {v}" for k, v in FAQ_DATA.items())
    prompt = (
        f"You are a helpful assistant. Here is a list of FAQs:\n\n{faq_text}\n\n"
        f"User question: \"{question}\". "
        f"Reply with the best match or say you don't know."
    )
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

# Wrap the tool
faq_tool = FunctionToolool(func=lookup_faq)

support_agent = LlmAgent(
    name="SupportAgent",
    description="An agent that answers users' questions based on a set of FAQs.",
    instruction="Use the FAQ tool to help answer customer questions.",
    model=AGENT_MODEL,
    tools=[faq_tool]
)

# Set up session service and runner
session_service = InMemorySessionService()
await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=support_agent, app_name=APP_NAME, session_service=session_service)

# Define and call the agent asynchronously
async def call_agent_async(query: str):
    print(f"\n>>> User Query: {query}")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break

    print(f"<<< Agent Response: {final_response_text}")

# Run the agent
await call_agent_async("What is your return policy?")



