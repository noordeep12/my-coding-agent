import os
import sys
import json
import argparse
from pathlib import Path

from my_coding_agent import Agent, tool, ToolsRegistry

parser = argparse.ArgumentParser(description="Agentic shell runner")
parser.add_argument("--prompt", "-p", type=str, default=None,
                    help="User message for the Main Agent (replaces the default task)")
parser.add_argument("--interactive", "-i", action="store_true",
                    help="Prompt interactively for the user message (paste mode, end with Ctrl+D)")
parser.add_argument("--discover", "-d", action="store_true",
                    help="Force-run the Discovery Agent even if .my_coding_agent/discovery.md already exists")
parser.add_argument("--analyze", "-a", action="store_true",
                    help="Run the Session Analyzer Agent after the main run (reads examples/stderr.log)")
parser.add_argument("--analyze-log", type=str, default="examples/stderr.log",
                    help="Log file for the Session Analyzer (default: examples/stderr.log)")
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



# ------------ DISCOVERY AGENT ----------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from agentic_discovery import run_discovery  # noqa: E402

run_discovery(force=args.discover)

tools = [
    tool(ToolsRegistry.bash),
    tool(ToolsRegistry.read_file),
    tool(ToolsRegistry.write_file),
]

# ------------ MAIN AGENT (runs the main task, with the discovery file as part of the system prompt) ------------
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use tools when needed. Use absolute paths when working with files. You are running in a Macbook Pro."
            "Available tools: "
            f"{'- ' + '\n- '.join([t['function']['name'] + ' function' + ' with function parameters ' + str(t['function']['parameters']['properties']) + ' that ' + t['function']['description'] for t in tools])}"
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

# ------------ SESSION ANALYZER AGENT -----------------------------------------
# Runs when: --analyze flag is set
if args.analyze:
    sys.path.insert(0, str(Path(__file__).parent))
    from agentic_session_analyzer import run_analysis  # noqa: E402
    run_analysis(log_path=args.analyze_log, max_steps=20)

# run python
# uv run python examples/agentic_shell.py > examples/stdout.log 2> examples/stderr.log
# run with discovery:  uv run python examples/agentic_shell.py --discover
# run with analysis:   uv run python examples/agentic_shell.py --analyze
# run both:            uv run python examples/agentic_shell.py --discover --analyze