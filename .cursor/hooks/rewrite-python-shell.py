"""preToolUse hook: rewrite bare `python`/`py` to rl_pytorch venv in Agent shell commands."""
from __future__ import annotations

import json
import re
import sys

VENV_PYTHON = r"E:\Python\venvs\rl_pytorch\Scripts\python.exe"
QUOTED_PYTHON = f'"{VENV_PYTHON}"'


def rewrite_command(command: str) -> str:
    if VENV_PYTHON.lower() in command.lower():
        return command

    updated = re.sub(r"\bpython\b(?!\.exe)", lambda _: QUOTED_PYTHON, command)
    updated = re.sub(r"\bpy\s+-", lambda _: f"{QUOTED_PYTHON} -", updated)
    updated = re.sub(r"\bpy\s+", lambda _: f"{QUOTED_PYTHON} ", updated)
    return updated


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Shell":
        print(json.dumps({"permission": "allow"}))
        return

    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command", "")
    rewritten = rewrite_command(command)

    if rewritten == command:
        print(json.dumps({"permission": "allow"}))
        return

    print(
        json.dumps(
            {
                "permission": "allow",
                "updated_input": {**tool_input, "command": rewritten},
            }
        )
    )


if __name__ == "__main__":
    main()
