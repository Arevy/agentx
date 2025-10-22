# Prompting Guidelines

## Prompt Structure
- The memory summary (`memory_bank/global.md`) is inserted at the top to provide persistent context.
- Relevant snippets from the local documentation (`docs/*.md`) selected by keyword matching follow next.
- The recent history (last 2–4 exchanges) is included for continuity.
- The prompt ends with `User: ...` and `Assistant:` to trigger response generation.

## Action Formats
- `Action: run_command("ls -la")` – executes the command inside a POSIX shell.
- `Action: open_file("path/to/file.py")` – reads the referenced file and returns the first 8,000 characters.
- `Action: store_memory("Summary of situation X")` – writes a timestamped note to the global memory.

## Good Practice
- Keep commands atomic and avoid dangerous chains (e.g., `rm`, `shutdown` are blocked).
- After each action, the model receives an `Observation: ...` message to continue its reasoning.
