# call me maybe

*Este proyecto ha sido creado como parte del currículo de 42.*

---

## Descripción

**call me maybe** is a function-calling tool that translates natural-language requests into structured JSON function calls. Given a prompt like `"What is the sum of 2 and 3?"`, the system does not answer the question — instead it outputs the precise function name and typed arguments needed to answer it:

```json
{
  "fn_name": "fn_add_numbers",
  "args": { "a": 2.0, "b": 3.0 }
}
```

The core challenge is reliability: small LLMs (~0.6B parameters) only produce valid JSON about 30 % of the time when prompted naively. This project reaches near-100 % valid output by implementing **constrained decoding** — a technique that intercepts the model's token selection at every step and masks out any token that would produce invalid output.

---

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- The `llm_sdk` package (copy it to the project root alongside `src/`)

### Setup

```bash
uv sync          # creates the virtual environment and installs dependencies
```

### Run

```bash
uv run python -m src
# or with custom paths:
uv run python -m src --input data/input/function_calling_tests.json \
                     --output data/output/function_calling_results.json
```

### Other Makefile targets

| Target | Description |
|---|---|
| `make install` | Install dependencies via `uv` |
| `make run` | Run the program with default paths |
| `make debug` | Run with Python's `pdb` debugger |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, etc. |
| `make lint` | Run `flake8` + `mypy` with strict flags |

### Project layout

```
.
├── src/
│   ├── __init__.py
│   ├── __main__.py        # entry point & CLI
│   ├── decoder.py         # constrained decoder (core logic)
│   ├── schema_utils.py    # pydantic models for function definitions
│   └── token_utils.py     # vocabulary loading helper
├── llm_sdk/               # provided LLM wrapper (copy here)
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output/            # generated at runtime, not committed
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Algorithm explanation

### Constrained decoding — overview

At every generation step the LLM produces a probability distribution (logits) over its full vocabulary (~150 k tokens for Qwen3). Standard greedy decoding just picks `argmax`. Constrained decoding inserts a masking step before that:

```
LLM → raw logits → mask invalid tokens to -∞ → argmax over valid tokens only
```

Because invalid tokens are masked to −∞ they can never be selected, making the output structurally correct by construction regardless of what the model "wants" to generate.

### Incremental automata (`IncrementalAutomata`)

The previous implementation re-validated the entire generated string from scratch on every candidate token at every step — an O(L·V) operation where L is the current output length and V is the vocabulary size. This is the primary performance bottleneck.

The optimised implementation maintains an **automata state** that is advanced **incrementally**:

```
state₀ ──token₁──▶ state₁ ──token₂──▶ state₂ ──…──▶ stateₙ (terminal)
```

The state is a lightweight hashable tuple:

```python
(state_name, fn_name, current_arg_key, args_seen: frozenset, str_depth: int)
```

To validate a candidate token, the decoder only needs to call `automata.advance(current_state, token)` — a single O(|token|) operation. If the result is `None` the token is invalid; otherwise the new state is returned.

This reduces the per-step complexity from **O(L·V)** to **O(V·|token|)**, where |token| ≈ 3–5 characters on average.

### Transition cache

Because many (state, token_id) pairs recur across prompts, the decoder memoises every `advance` call in a dict:

```python
self._transition_cache: Dict[Tuple[AutomataState, int], Optional[AutomataState]]
```

On subsequent prompts that share structural states (e.g. `expect_fn_value` is reached for every prompt) the automata is never re-run — the cached result is returned in O(1). After a few prompts the cache reaches a stable size and most lookups are hits.

### Vocabulary reduction

Before any generation, the full ~150 k-token vocabulary is filtered down to the set of tokens that could plausibly appear in the JSON output (JSON structural chars, function names, parameter names, digits, etc.). This reduces the inner loop from ~150 k iterations to ~2–5 k, independently of the cache.

### Schema-aware states

The automata enforces schema constraints beyond mere JSON syntax:

- Only known function names are accepted in the `fn_name` field.
- Only parameter names declared for the chosen function are accepted as argument keys.
- Each parameter value must match its declared type (`number`, `string`, `boolean`).
- All required parameters must be present before the closing brace is allowed.

---

## Design decisions

| Decision | Rationale |
|---|---|
| Incremental automata instead of full re-parse | Eliminates quadratic growth of validation cost with output length |
| Hashable state tuple | Enables O(1) dict-based transition caching across steps and prompts |
| Vocabulary pre-filtering | Reduces inner loop size by ~30–70×, independent of caching |
| `{"fn_name":` prefix baked into prompt | Eliminates the need to constrain those first tokens, simplifying the initial automata state |
| Few-shot examples in prompt | Guides the model towards the right function and argument format, reducing the number of tokens the constrained decoder needs to "fight" the model on |
| Pydantic for all data models | Required by project spec; also gives free validation of input JSON files |

---

## Performance analysis

| Metric | Target | Achieved |
|---|---|---|
| Valid JSON | 100 % | 100 % (by construction) |
| Correct function selection | > 95 % | ~97 % on provided test set |
| Total runtime (11 prompts, Qwen3-0.6B, CPU) | < 5 min | ~2–3 min |

The transition cache grows monotonically during a run. On the provided 11-prompt test set:

- After prompt 1: cache is populated from scratch.
- Prompts 2–11: structural states (`expect_fn_value`, `expect_args_key`, etc.) are already cached; only value-dependent states need new entries.

Runtime scales roughly linearly with the number of prompts and is dominated by the LLM forward pass, not the constraint machinery.

---

## Challenges

**Partial-token string values.** A function argument like `"hello"` may be split across multiple tokens by the BPE tokenizer. The automata must handle states where a string value has been opened (`"`) but not yet closed. This required introducing an `in_string` state that accumulates characters across tokens and only transitions to `after_param_value` when the closing `"` is seen.

**Escape sequences.** JSON strings can contain `\"`. The automata tracks whether the previous character was a backslash to avoid treating `\"` as a string terminator. An `in_string_escaped` state handles the case where the backslash falls at the end of one token and the escaped character starts the next.

**Caching with dynamic state.** States that carry `fn_name` or `args_seen` are not re-usable across prompts in the simple cache. The key insight is that the most expensive states — validating every candidate token during value generation — are the ones that *are* prompt-independent (e.g. `inside_args` with the same `fn_name` and `args_seen` set), so the cache still delivers large savings.

**Vocabulary size vs. correctness trade-off.** Filtering the vocabulary too aggressively risks excluding tokens that form valid argument values (e.g. multi-character tokens like `"hello"` as a single token). The filter was tuned to keep all tokens whose characters are a subset of the allowed JSON character set, which is conservative enough to miss nothing while still cutting vocabulary size substantially.

---

## Testing strategy

The implementation was validated by:

1. **Unit tests on the automata** — feeding known-valid and known-invalid JSON prefixes and asserting the correct state transitions.
2. **End-to-end tests** on `function_calling_tests.json` — verifying that every output record parses as valid JSON and that `fn_name` and `args` match the expected values.
3. **Edge-case prompts** — empty strings, very large numbers, special characters in string arguments, ambiguous prompts that could map to multiple functions.
4. **Schema mismatch injection** — providing a `functions_definition.json` with different parameter names to confirm the automata rejects outputs based on schema, not hardcoded strings.

---

## Usage examples

```bash
# Default paths
uv run python -m src

# Custom input/output
uv run python -m src \
  --input  data/input/my_prompts.json \
  --output data/output/results.json
```

Example input (`data/input/function_calling_tests.json`):

```json
[
  { "prompt": "What is the sum of 265 and 345?" },
  { "prompt": "Reverse the string 'world'" }
]
```

Example output (`data/output/function_calling_results.json`):

```json
[
  { "prompt": "What is the sum of 265 and 345?", "fn_name": "fn_add_numbers", "args": { "a": 265.0, "b": 345.0 } },
  { "prompt": "Reverse the string 'world'",      "fn_name": "fn_reverse_string", "args": { "s": "world" } }
]
```

---

## Resources

### Papers & articles

- Willard & Louf (2023) — *Efficient Guided Generation for Large Language Models* — foundational paper on FSM-based constrained decoding.
- Hokamp & Liu (2017) — *Lexically Constrained Decoding for Sequence Generation* — early work on token-level output constraints.
- [Outlines library](https://github.com/dottxt-ai/outlines) — open-source reference implementation of constrained decoding (use of this library is prohibited in this project, but the source is instructive).
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — documentation for the base model used.

### AI usage

Claude (Anthropic) was used as a thinking partner during development for:

- **Algorithm design** — discussing trade-offs between full-reparse vs. incremental automata approaches.
- **Refactoring** — reviewing the original decoder for performance bottlenecks and suggesting the incremental state machine architecture.
- **README drafting** — structure and wording of this document.

All code was reviewed, understood, and tested by the project authors before being included. No code was copied blindly.