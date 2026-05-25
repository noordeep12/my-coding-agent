import os
import json
import argparse

from my_coding_agent import LLM, Agent, tool, ToolsRegistry

parser = argparse.ArgumentParser(description="Agentic shell runner")
parser.add_argument("--prompt", "-p", type=str, default=None, help="User message for the Main Agent (replaces the default task)")
parser.add_argument("--interactive", "-i", action="store_true", help="Prompt interactively for the user message (paste mode, end with Ctrl+D)")
args = parser.parse_args()

DEFAULT_PROMPT = "Using `git` and `gh` CLI tools, ensure the latest local code changes is committed and pushed to GitHub, with standardized commit messages."

if args.interactive:
    print("Enter your prompt (paste content freely, press Ctrl+D when done):")
    print("─" * 60)
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    user_prompt = "\n".join(lines).strip()
    if not user_prompt:
        user_prompt = DEFAULT_PROMPT
    print("─" * 60)
    print()
elif args.prompt is not None:
    user_prompt = args.prompt
else:
    user_prompt = DEFAULT_PROMPT

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


# if discovery file already exists, skip discovery step
if not os.path.isfile(".my_coding_agent/discovery.md"):
    print("Discovery file does not exist. Creating it.")

    discovery_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use tools when needed. Use absolute paths when working with files. You are running in a Macbook Pro."
                "Available tools: "
                "- bash(command) - executes a bash command and returns its output. "
                "Workspace:"
                f"- Current path: {os.getcwd()} "
                f"- Current directory contents: {os.listdir(os.getcwd())} "
                f"- Current OS: {os.name}, Platform: {os.sys.platform}, User: {os.getlogin()}"
                f"- Git status: {os.popen('git status').read() if os.path.isdir('.git') else 'Not a git repository'}"
                f"- Git branch: {os.popen('git rev-parse --abrev-ref HEAD').read().strip() if os.path.isdir('.git') else 'Not a git repository'}"
                f"- Git recent commits: {os.popen('git log -5 --oneline').read() if os.path.isdir('.git') else 'Not a git repository'}"
            )
        },
        {
            "role": "user", 
            "content": "Explore the workspace and discover any relevant information that can help you understand the current state of the codebase, git repository, recent activities, and anything else that might be useful for future tasks. Summarize your findings in a concise way into a Markdown file saved to `.my_coding_agent/discovery.md` in the current workspace. It should contains: 1. a summary of the current state of the codebase and git repository, 2. any recent activities or changes that might be relevant, 3. any insights or observations that can help you understand the context better. 4. Tool descriptions and how to use them. 5. Latest user requests. 6. Recent transcript of the conversation. The goal is to create a Stable prompt prefix that can be prepended to future conversations to provide context and continuity, even if the conversation history gets truncated due to token limits."
        }
    ]


    # Agent that discover the current workspace
    tools = [
        tool(ToolsRegistry.bash),
        tool(ToolsRegistry.read_file),
        tool(ToolsRegistry.write_file),
    ]
    agent = Agent(
        messages=discovery_messages,
        tools=tools,
        label="Discovery Agent",
    )
    discovery_messages = agent.run(max_steps=20)


# test discovery file content
print("Discovery file content:")
try:    
    with open(".my_coding_agent/discovery.md", "r") as f:
        print(f.read())
except Exception as e:    
    print(f"Error reading discovery file: {e}")   


# import ipdb; ipdb.set_trace()

# Main Agent
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use tools when needed. Use absolute paths when working with files. You are running in a Macbook Pro."
            "Available tools: "
            "- bash(command) - executes a bash command and returns its output. "
            "Workspace:"
            f"- Current path: {os.getcwd()} "
            f"- Current directory contents: {os.listdir(os.getcwd())} "
            f"- Current OS: {os.name}, Platform: {os.sys.platform}, User: {os.getlogin()}"
            f"- Git status: {os.popen('git status').read() if os.path.isdir('.git') else 'Not a git repository'}"
            f"- Git branch: {os.popen('git rev-parse --abrev-ref HEAD').read().strip() if os.path.isdir('.git') else 'Not a git repository'}"
            f"- Git recent commits: {os.popen('git log -5 --oneline').read() if os.path.isdir('.git') else 'Not a git repository'}"
            f"- Discovery notes: {os.popen('cat .my_coding_agent/discovery.md').read() if os.path.isfile('.my_coding_agent/discovery.md') else 'No discovery notes found'}"
        )
    },
    {
        "role": "user", 
        "content": user_prompt
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
    label="Main Agent",
)
messages = agent.run(max_steps=20)
print("Messages: ", json.dumps(messages, indent=4))

# run python
# uv run python examples/agentic_shell.py > examples/stdout.log 2> examples/stderr.log