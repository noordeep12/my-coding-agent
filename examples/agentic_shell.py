import os
import json

from my_coding_agent import LLM, Agent, tool, ToolsRegistry

# disabled
# cisa_kev_demo = [
#     {
#         "role": "system",
#         "content": (
#             "You are a helpful assistant that has terminal bash access. Use absolute paths when working with files. You are running in a Macbook Pro."
#             "Available tools: "
#             "* bash(command) - executes a bash command and returns its output. "
#             f"Current path: {os.getcwd()} "
#             f"Current directory contents: {os.listdir(os.getcwd())} "
#             f"Current OS: {os.name}, Platform: {os.sys.platform}, User: {os.getlogin()}"
#         )
#     },
#     {
#         "role": "user", 
#         "content": "Using `python` command, collect the latest vulnerability CISA KEV data of today and save it to a file named `examples/cisa_kev.json` in the current directory. Then read the file and return the content."
#     }
# ]

# disabled update readme
# messages = [
#     {
#         "role": "system",
#         "content": (
#             "You are a helpful assistant. Use tools when needed. Use absolute paths when working with files. You are running in a Macbook Pro."
#             "Available tools: bash(command) - executes a bash command and returns its output. "
#             f"Current path: {os.getcwd()} "
#             f"Current directory contents: {os.listdir(os.getcwd())} "
#             f"Current OS: {os.name}, Platform: {os.sys.platform}, User: {os.getlogin()}"
#         )
#     },
#     {
#         "role": "user", 
#         "content": "Please update the README.md file in the current directory to include a new section about this agentic shell demo, with instructions on how to run it and what it does. If the README.md file does not exist, create one with the necessary content."
#     }
# ]

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use tools when needed. Use absolute paths when working with files. You are running in a Macbook Pro."
            "Available tools: bash(command) - executes a bash command and returns its output. "
            f"Current path: {os.getcwd()} "
            f"Current directory contents: {os.listdir(os.getcwd())} "
            f"Current OS: {os.name}, Platform: {os.sys.platform}, User: {os.getlogin()}"
        )
    },
    {
        "role": "user", 
        "content": "Using `git` and `gh` CLI tools, ensure the latest local code changes is committed and pushed to GitHub, with standardized commit messages."
    }
]



tools = [
    tool(ToolsRegistry.bash),
    tool(ToolsRegistry.read_file),
    tool(ToolsRegistry.write_file),
]

print("Initial messages: ", json.dumps(messages, indent=4))
print("Available tools: ", json.dumps(tools, indent=4))
print("")
print("")

agent = Agent(
    messages=messages,
    tools=tools,
)

final_messages = agent.run(max_steps=20)
print("Final messages: ", json.dumps(final_messages, indent=4))