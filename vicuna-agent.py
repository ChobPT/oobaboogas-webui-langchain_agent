from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.utilities import WikipediaAPIWrapper


import requests



wikipedia = WikipediaAPIWrapper()


context= "Answer the following questions as best you can. You only have access to the following tools:\
\
Wikipedia: Gives access to information about current and past topics, people, events, concepts, explanations \
\
Use the following format: \
\
Question: the input question you must answer\
Thought: you should always think about what to do\
Action: the action to take, should be one of [Python REPL]\
Action Input: the input to the action\
Observation: the result of the action\
... (this Thought/Action/Action Input/Observation can repeat N times)\
Thought: I now know the final answer\
Final Answer: the final answer to the original input question \
\
Begin!\
\
Question: "

class webuiLLM(LLM):        
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            "http://localhost:5000/api/v1/generate",
            json={
                'prompt':prompt,
                'max_new_tokens': 200,
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
      
        return response.json()["results"][0]["text"].strip().replace("```", " ").replace("\\n", "\\ \\n ")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {

        }
    



# from alpaca_request_llm import AlpacaLLM

# First, let's load the language model we're going to use to control the agent.

#llm = CustomLLM()

llm = webuiLLM()
# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
#tools = load_tools(['python_repl'], llm=llm)
tools = [
    Tool(
        name='wikipedia',
        func= wikipedia.run,
        description="Useful for when you need to find information about a topic, country or person"
    )
]
# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#Now let's test it out!
orders = input("Orders? ")

agent.run(context+orders)    
