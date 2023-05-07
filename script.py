import asyncio
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.agents import AgentType, Tool, initialize_agent,load_tools
from typing import Any, Dict, List, Optional, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult

#tools
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper


from modules import shared as shared
from modules import chat as chat
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from modules.text_generation import (encode, generate_reply,
                                     stop_everything_event)
from langchain.callbacks.base import BaseCallbackManager, AsyncCallbackHandler
import base64
import re
import time
from dataclasses import dataclass
from functools import partial
from io import BytesIO

import requests
import json
from pathlib import Path


import requests


def output_modifier(string):

    return string


def sendprompt(texttosend):
    print("[DEBUG]Sending Prompt Chat...")
    return chat.send_dummy_message(texttosend, shared.gradio["name1"],shared.gradio["name2"],shared.gradio['mode'])
def sendchat(texttosend):
    print("[DEBUG]Sending Chat...")
    return chat.send_dummy_reply(texttosend, shared.gradio["name1"],shared.gradio["name2"],shared.gradio['mode'])
    


context = "Answer the following questions as best you can. You only have access to the following tools:\
\
Wikipedia: Gives access to information about current and past topics, people, events, concepts, explanations \
\
Use the following format: \
\
Question: the input question you must answer\
Thought: you should always think about what to do\
Action: the action to take, should be one of [wikipedia]\
Action Input: the input to the action\
Observation: the result of the action\
... (this Thought/Action/Action Input/Observation can repeat N times)\
Thought: I now know the final answer\
Final Answer: the final answer to the original input question \
\
Begin!\
\
Question: "


class WebAgentCallBack(AsyncCallbackHandler):
    """Custom CallbackHandler."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        print("\n\n\033[1m> Prompts:\033[0m")
        pass

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        sendchat("[DEBUG]LLM End...")
        print(response)

        pass

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")
        
        return sendprompt(f">> Entering new {class_name} chain...\033[0m")

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\n\033[1m> Finished chain.\033[0m")
        return sendchat(outputs)

    async def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        print("\n\033[1m> Error in chain.\033[0m")
        sendchat(str(self))
        output_modifier(">> Error in chain.")
        pass

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""
        print("[DEBUG]Using Tool...")
        pass

    async def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        sendchat("[DEBUG]Action Running...")
        print("[DEBUG]Action Running...")

    async def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        sendchat("[DEBUG]Tool Finished...")
        print(output)

    async def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    async def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        """Run when agent ends."""
        print("[DEBUG]Let there be text ...")
        print(text)

    async def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        
        print("[DEBUG]Agent Finished finishing...")
        output_modifier(finish.return_values)
        print("[DEBUG]Agent Finished...")
        


manager = BaseCallbackManager([WebAgentCallBack()])


class webuiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            "http://localhost:5000/api/v1/generate",
            json={
                'prompt': prompt,
                'max_new_tokens': 100,
                'do_sample': False,
                'temperature': 0.7,
                'top_p': 0.1,
                'typical_p': 1,
                'repetition_penalty': 1.18,
                'top_k': 40,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'num_beams': 1,
                  'penalty_alpha': 0,
                'length_penalty': 1,
                'early_stopping': True,
                'seed': -1,
                'add_bos_token': True,
                'truncation_length': 2048,
                'ban_eos_token': False,
                'skip_special_tokens': False,
                'stopping_strings': ["\n\n", "Observation:"]
            }
        )

        response.raise_for_status()

        return response.json()["results"][0]["text"].strip().replace("```", " ")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {

        }



wikipedia = WikipediaAPIWrapper()

llm = webuiLLM()
# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(['wikipedia'], llm=llm)
tools = [
    Tool(
        name='wikipedia',
        func=wikipedia.run,
        description="Useful for when you need to find information about a topic."
    )
]
# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent =  initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, callback_manager=manager, verbose=True)


def input_modifier(string):
    if string[:3] == "/do":
        agent.run(context+string)
    else:
        output_modifier(string.split("###")[0].split("Human:")[0])
    return string.replace('/do ', '')
