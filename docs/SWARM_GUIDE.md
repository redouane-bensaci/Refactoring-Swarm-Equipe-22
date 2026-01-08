# Refactoring Swarm – Newbie Guide

This repo implements a **3‑agent refactoring loop** (Auditor → Fixer → Judge) coordinated by **LangGraph**.

The goal: given a target (a directory of Python files or a single `.py` file), the swarm iterates until tests pass (or it hits limits / rate limits).

---

## 1) Important Files (where to look first)

- src/swarm_orchestrator.py
  - Owns the **state machine** and all **stop conditions**.
  - Defines the swarm state (`SwarmState`), agent functions (Auditor/Fixer/Judge), file queueing, and LangGraph routing.

- src/services/llm_client.py
  - Owns **model/provider construction** (`get_llm`) and provider detection (`get_model_config`).

- src/services/test_runner.py
  - Owns **pytest execution** and converting results into a structured dict.
  - Produces human-readable summaries used by Judge.

- src/services/file_handler.py
  - Safe read/write/list operations inside the target directory.

- src/services/pylint_tool.py
  - Runs pylint and produces summaries consumed by Auditor.

- main.py
  - CLI entrypoint: parses args, validates the target, runs the swarm, sets exit codes.

- logs/experiment_data.json
  - Append-only log of what happened per agent/tool call.

---

## 2) How to Run

Directory mode:

- `python main.py --target sandbox/test_code --max-iterations 30`

Single-file mode:

- `python main.py --target sandbox/test2/buggy_stats.py --max-iterations 10`

Notes:

- The CLI argument is defined as `--target_dir`, but `argparse` allows `--target` as an abbreviation.
- If OpenRouter is rate-limited, the swarm may stop early (see “Stop reasons”).

---

## 3) Model / Provider Configuration

### 3.1 Where the model is created

In src/services/llm_client.py:

- `get_llm(model_name, temperature, max_tokens, api_key=None)` creates the LangChain chat model instance.

Providers:

- OpenRouter (via `langchain_openai.ChatOpenAI`) when `OPENROUTER_API_KEY` is set (or an explicit `api_key` is passed).
- Google AI (via `langchain_google_genai.ChatGoogleGenerativeAI`) when `GOOGLE_API_KEY` is set.

In src/swarm_orchestrator.py:

- `create_swarm()` calls `get_llm(model_name=..., api_key=get_openrouter_api_key())`.
- `get_openrouter_api_key()` currently enforces **OPENROUTER_API_KEY2**.

### 3.2 What each LLM parameter means

In `get_llm(...)`:

- `model_name`
  - Provider-specific model identifier (e.g., `meta-llama/llama-3.2-3b-instruct:free`).

- `temperature`
  - Randomness of sampling. Lower is more deterministic.

- `max_tokens`
  - Upper bound for output length. Larger = longer answers but higher cost/latency.

- `api_key`
  - If supplied, it overrides env vars. Otherwise `OPENROUTER_API_KEY` is used.

### 3.3 API keys (.env)

You typically set:

- `OPENROUTER_API_KEY2=...` (required by `get_openrouter_api_key()`)

Optional:

- `OPENROUTER_API_KEY=...` (used by `llm_client.get_llm` if key2 isn’t enforced)
- `GOOGLE_API_KEY=...`

---

## 4) How Agents Communicate (SwarmState)

Agents don’t call each other directly.

They **communicate only by returning updates to a shared dict** (LangGraph state).

In src/swarm_orchestrator.py, the state is:

- `active_agent`: which node should run next (`"auditor"`, `"fixer"`, `"judge"`, `"done"`).

- `target_path`: what we’re refactoring/testing (directory or file).

- `pending_files`: queue of module file paths when `target_path` is a directory.

- `current_file`: the file currently being processed from the queue.

- `iteration_count`: incremented by the router each step.

- `max_iterations`: your CLI cap.

- `stop_reason`: why the swarm ended (e.g., `"tests_passing"`, `"rate_limited"`).

- `messages`: append-only list of human-readable traces (prompts, summaries, decisions).

---

## 5) The 3 Agents (what each does)

### 5.1 Auditor

Created by `create_auditor(llm)`.

What it does:

- Runs pylint via `analyze_code_with_pylint(target_path)`.
- Appends a summary to `messages`.
- Sets `active_agent = "fixer"`.

Important detail:

- Auditor currently limits analysis to the first 5 Python files found.

### 5.2 Fixer

Created by `create_fixer(llm)`.

What it does:

- Chooses which file to work on:
  - If `current_file` exists, it uses that.
  - Else it pulls the next file from `pending_files`.

- Refactors the code using the LLM (with backoff):
  - `invoke_with_backoff(llm, prompt)` retries short on 429/rate-limit errors.

- Writes fixed code via `FileHandler.write_file(..., backup=True)`.

- Test generation / repair strategy:
  - If pytest collected 0 tests, bootstrap a minimal `test_<module>.py`.
  - If pytest has a **collection error**, deterministically rewrite the per-module test file based on the module’s AST (avoids placeholder imports like `module_under_test`).
  - If a test file already exists and is syntactically valid, Fixer will not overwrite it.
  - If LLM-generated tests don’t parse, Fixer refuses to write them.

### 5.3 Judge

Created by `create_judge(llm)`.

What it does:

- Runs pytest via `validate_with_tests(target_path)`.
  - If `target_path` is a file, pytest is run in the file’s parent directory.

- Converts raw results into a summary string using `get_test_summary(...)`.

- Decides the next node:
  - If ✅ and there are still queued files → continue (Fixer processes next file).
  - If ✅ and queue is empty → `active_agent = "done"`.
  - Otherwise → `active_agent = "auditor"`.

---

## 6) Routing / Stop Conditions (LangGraph)

In src/swarm_orchestrator.py:

- A `StateGraph(SwarmState)` is built.
- Nodes are `auditor`, `fixer`, `judge`.
- A routing function `route_to_next(state)`:
  - Prints progress.
  - Stops when `iteration_count >= max_iterations`.
  - Stops when `active_agent == "done"`.
  - Otherwise increments `iteration_count` and returns the next node name.

LangGraph recursion limit:

- LangGraph enforces a step recursion limit.
- `run_swarm()` passes `config={"recursion_limit": max(100, max_iterations * 10)}` to avoid `GraphRecursionError` before our own stop conditions apply.

---

## 7) Test Runner Details

In src/services/test_runner.py:

- Primary attempt uses `--json-report` (requires `pytest-json-report`).
- If that plugin is missing, it retries without JSON flags.

Special conditions:

- `no_tests_collected`
  - Detects pytest exit code 5 or common output strings.

- `collection_error`
  - Detects failures during collection/import (tests can’t start).

Those are important because “0/0” results can be misleading without this distinction.

---

## 8) Exit Codes (CLI)

In main.py:

- Exit `0`: swarm ended due to `stop_reason == "tests_passing"`.
- Exit `1`: swarm failed or never reached a passing test state.
- Exit `2`: swarm stopped due to LLM rate limits.

---

## 9) Common Newbie Gotchas

- Import-time side effects
  - A module that runs code on import will fail `test_imports.py` (and will break collection).
  - Fix by guarding with `if __name__ == "__main__":`.

- Placeholder test imports
  - LLMs often emit `from module_under_test import ...` or `from your_module import ...`.
  - The Fixer now repairs these on collection errors.

- Rate limits
  - If OpenRouter returns 429s, the swarm stops early to avoid thrashing.

---

## 10) Where to Extend Next

If you want the swarm to be stricter for full directories:

- Add per-file “done” criteria (e.g., require import smoke + targeted unit tests).
- Add a “stuck detector” (same Judge result N times → stop and report).
- Add a consistent file ordering / prioritization strategy.
