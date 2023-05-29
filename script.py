import asyncio, datetime,base64,re,time,requests,json, os, re
from typing import Optional, List, Mapping, Any, Union
from modules import shared as shared
from modules import chat as chat
from modules.extensions import apply_extensions
from modules.text_generation import encode, get_max_prompt_length
from modules.text_generation import (encode, generate_reply,
                                     stop_everything_event)

from dataclasses import dataclass
from functools import partial
from io import BytesIO

from langchain.llms.base import LLM

from langchain.agents import AgentType, Tool, initialize_agent,load_tools
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.callbacks.base import BaseCallbackManager, AsyncCallbackHandler, BaseCallbackHandler
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain

from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import BaseTool

#import the langchain wikipedia wrapper
from langchain.utilities import WikipediaAPIWrapper

from pathlib import Path


os.environ["OPENAI_API_TYPE"] = "open_ai"
os.environ["OPENAI_API_KEY"] = "123"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:5001/v1"
#change the environ openedai_debug to true
os.environ["OPENAI_DEBUG"] = str("true")



def output_modifier(string):
    return string


def sendprompt(texttosend):
    print("[DEBUG]Sending Prompt Chat...")
    return chat.send_dummy_message(texttosend)
def sendchat(texttosend):
    print("[DEBUG]Sending Chat...")
    return chat.send_dummy_reply(texttosend)


template = """USER:Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input  or the final conclusion to your thoughts


Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
ASSISTANT: {agent_scratchpad}"""
searchWrapper=WikipediaAPIWrapper()



tools = [
    Tool(
        name = "Search",
        func=searchWrapper.run,
        description="Wikipedia serves as a versatile tool, offering uses such as gathering background information, exploring unfamiliar topics, finding reliable sources, understanding current events, discovering new interests, and obtaining a comprehensive overview on diverse subjects like historical events, scientific concepts, biographies of notable individuals, geographical details, cultural phenomena, artistic works, technological advancements, social issues, academic subjects, making it a valuable resource for learning and knowledge acquisition."
    )
]

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += "\Thought:"+action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n Thought: ".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

#Try to ask a question to openai through langchain


def split_text(text):
    blocks = text.split("Page:")
    print('blocks', blocks)
    if(len(blocks) < 1):
        if(len(blocks) < 2):
            first_block = blocks[0].strip()+'\n Page: '+blocks[1].strip()  # Get the first block and remove leading/trailing whitespace
        else:
            first_block = blocks[0].strip
            print('first_block', first_block)
    else:
        first_block = text
        print('first_block', first_block)
    return first_block

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        llm_output = split_text(str(llm_output))
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            #raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[0].strip()},
                log=llm_output,
            )
        else:
            action = match.group(1).strip()
            action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()


llm = OpenAI(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def input_modifier(string):
    if string[:3] == "/do":
       agent_executor.run(string)
    else:
        output_modifier(string.split("###")[0].split("Human:")[0])
    return string.replace('/do ', '')
