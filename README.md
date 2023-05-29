# oobaboogas-webui-langchain_agent

Creates an Langchain Agent which uses the WebUI's API and Wikipedia to work. 

This has been reworked to use the openAI API on the Oobabooga's extension, and requirements have been dramatically shrunk down, generated with pipreqs

Tested to be working, I learned python a couple of weeks ago, bear with me.

Needs the  `openai` and `no_stream` extensions enabled enabled. (` --extensions openai --no-stream` added when running the WebUI)
![Screenshot 2023-05-29 11 50 41 PM](https://github.com/ChobPT/oobaboogas-webui-langchain_agent/assets/45816945/194dc79a-c44e-43a6-a9c3-e9505d7b2613)
![Screenshot 2023-05-29 11 49 56 PM](https://github.com/ChobPT/oobaboogas-webui-langchain_agent/assets/45816945/57314a63-cffc-45a1-87e2-5b7e0e0ffdc6)


Install with `pip install -r requirements.txt` 

### Installation

go to the WebUI folder and   
```bash
cd extensions;
git clone https://github.com/ChobPT/oobaboogas-webui-langchain_agent/ webui_langchain_agent;
cd webui_langchain_agent;
pip install -r requirements.txt;
cd ../..
```

### Usage
To trigger simply add /do before the instructions so that you can continue the conversation later on with the context

You can basically enable the tools by just using the documentation basics at https://python.langchain.com/en/latest/modules/agents/tools.html and then add the respective tools at

![image](https://user-images.githubusercontent.com/45816945/236650063-3220e6f6-5ce9-40b7-8252-43d2cab3ac87.png)


Tested with TheBloke_airoboros-13B-GPTQ

### RoadMap

Development will be slow! This is something that is worked on on the (rare) spare time, contributions are welcome.

Again, just learned python recently, it kind of works, but it's a start

cheers!
