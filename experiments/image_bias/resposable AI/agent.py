#!pip install protobuf==5.28.1 google-adk==1.0.0 litellm -q -q
#pip install Deprecated
import importlib
importlib.invalidate_caches()

import os
os.environ["OPENAI_API_BASE"]="http://localhost:11434/v1"

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

AGENT_MODEL = LiteLlm(model="openai/gpt-4o-mini")

agent = LlmAgent(
    name="WelcomeAgent",
    description="Always greet the user politely",
    instruction="____",
    model=AGENT_MODEL
)

print(f"Agent '{agent.name}' created.")