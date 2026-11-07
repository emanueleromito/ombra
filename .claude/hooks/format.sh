#!/usr/bin/env bash
# PostToolUse hook: format the file an agent just edited.
# Reads the hook JSON on stdin, formats C++/CUDA with clang-format and Python with ruff.
# Never blocks the agent: missing tools or format errors are ignored.

input=$(cat)
file=$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))' 2>/dev/null)

[ -n "$file" ] && [ -f "$file" ] || exit 0

case "$file" in
  *.cpp|*.hpp|*.h|*.cc|*.cu|*.cuh)
    command -v clang-format >/dev/null 2>&1 && clang-format -i "$file" 2>/dev/null
    ;;
  *.py)
    command -v ruff >/dev/null 2>&1 && ruff format --quiet "$file" 2>/dev/null
    ;;
esac

exit 0
