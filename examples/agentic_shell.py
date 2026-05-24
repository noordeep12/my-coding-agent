import importlib
import my_coding_agent

importlib.reload(my_coding_agent)

from my_coding_agent import LLM, Agent, tool, ToolsRegistry
from my_coding_agent.llm import OMLX_API_URL, OMLX_API_KEY, OMLX_MODEL
from my_coding_agent.utils import extract_message, extract_finish_reason


import subprocess
import os
import json

from my_coding_agent._logging import get_logger
from my_coding_agent.tools import ToolsRegistry, tool


def bash(command: str) -> str:
    """
    Execute a bash command and return its output.
    When working with paths, use absolute paths to avoid issues with the current working directory.
    """
    cwd = os.getcwd()
    command = command.strip()
    result = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        env=os.environ,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    # decode result stdout
    try:
        stdout = result.stdout.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception as e:
        stdout = result.stdout

    # decode result stderr
    try:
        stderr = result.stderr.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception as e:
        stderr = result.stderr
    
    output = {
        "stdout": stdout, 
        "returncode": result.returncode, 
        "stderr": stderr
    }
    return  json.dumps(output, indent=4)

# Attach to registry so the LLM can call it
ToolsRegistry.bash = staticmethod(bash)

# Execution

cisa_kev_demo = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant that has terminal bash access. Use absolute paths when working with files. You are running in a Macbook Pro."
            "Available tools: "
            "* bash(command) - executes a bash command and returns its output. "
            f"Current path: {os.getcwd()} "
            f"Current directory contents: {os.listdir(os.getcwd())} "
            f"Current OS: {os.name}, Platform: {os.sys.platform}, User: {os.getlogin()}"
        )
    },
    {
        "role": "user", 
        "content": "Using `python` command, collect the latest vulnerability CISA KEV data of today and save it to a file named `examples/cisa_kev.json` in the current directory. Then read the file and return the content."
    }
]


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