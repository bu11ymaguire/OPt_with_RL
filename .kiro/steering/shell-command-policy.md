---
inclusion: always
---

# PowerShell command construction

Kiro's Windows shell permission parser must be able to classify every command.
Follow these rules for all terminal tool calls in this workspace:

- Run exactly one simple command per terminal tool call.
- Do not join commands with `;`, `&&`, `||`, or `|`.
- Do not begin a command with a PowerShell variable assignment such as `$py =`
  or `$path =`, and do not set `$env:` variables inline.
- Do not use multiline `python -c`, here-strings, command substitutions, or
  redirection in a terminal tool call.
- Never invoke an absolute Windows executable path as a bare command. PowerShell
  executable paths must use the call operator and double quotes, in the exact
  form `& "C:\absolute\path\program.exe" arguments`.
- Programs located below
  `C:\Users\jwkim\OneDrive\Desktop\OPt_with_RL` may be executed without
  additional consent when invoked with `&` and their quoted absolute path.
- Invoke the experiment Python interpreter in the exact form
  `& "C:\Users\jwkim\.venvs\opt-with-rl\Scripts\python.exe" arguments`.
  For example, run tests as
  `& "C:\Users\jwkim\.venvs\opt-with-rl\Scripts\python.exe" -m pytest tests/ -q`.
  It may also be launched with `Start-Process` for a long-running experiment.
- Deletion of generated files and directories is allowed below
  `C:\Users\jwkim\OneDrive\Desktop\OPt_with_RL`. Always use a resolved absolute
  path beginning with that exact root. Never use a relative deletion target,
  wildcard that can escape the root, `$HOME`, `~`, or another computed root.
- Before a recursive deletion, verify the resolved absolute target remains
  below the OPT_with_RL root, then run one canonical `Remove-Item` command with
  the absolute target.
- If a multi-step script is necessary, create a small script with the file
  editing tool, execute it using `&` plus its quoted absolute path, and remove
  the temporary file afterward.
- Use file editing tools for source changes instead of PowerShell text-rewrite
  one-liners.
- Run waiting, process inspection, and log inspection as separate commands.
