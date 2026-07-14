import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from .schema_utils import FunctionDefinition

_JSON_PREFIX = '{"fn_name":"'


class ConstrainedDecoder:
    def __init__(
        self,
        model: Any,
        vocab: Dict[int, str],
        functions: List[FunctionDefinition],
        verbose: bool = False,
        verbose_callback: Optional[Callable[[str], None]] = None,
        cache_size: Optional[int] = 100,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.functions = functions
        self.verbose = verbose
        self._verbose_callback: Callable[[str], None]

        if verbose_callback is not None:
            self._verbose_callback = verbose_callback
        else:
            def _default_verbose(msg: str) -> None:
                if verbose:
                    sys.stderr.write(msg + "\n")

            self._verbose_callback = _default_verbose

        self._fn_by_name: Dict[str, FunctionDefinition] = {
            f.name: f for f in functions
        }
        self._static_prompt: str = self._build_static_prompt()
        self._prefix_ids: List[int] = self._encode_static_prompt()

        self._cache: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self._cache_size = cache_size

    def encode(self, text: str) -> List[int]:
        """Expone la codificación del tokenizer subyacente."""
        return [int(x) for x in self.model.encode(text)[0].tolist()]

    def decode(self, token_ids: List[int]) -> str:
        """Decodifica una lista de IDs a texto usando el vocabulario."""
        return "".join(self.vocab.get(tid, "") for tid in token_ids)

    def _build_static_prompt(self) -> str:
        func_lines = [
            f"- {f.name}({', '.join(f.parameters.keys())})"
            for f in self.functions
        ]
        examples: List[Tuple[str, str]] = [
            (
                "What is the sum of 2 and 3?",
                '{"fn_name": "fn_add_numbers", "args": {"a": 2.0, "b": 3.0}}',
            ),
            (
                "Greet Maria",
                '{"fn_name": "fn_greet", "args": {"name": "Maria"}}',
            ),
            (
                'Reverse the string "hello"',
                '{"fn_name": "fn_reverse_string", "args": {"s": "hello"}}',
            ),
            (
                "What is the square root of 16?",
                '{"fn_name": "fn_get_square_root", "args": {"a": 16.0}}',
            ),
            (
                'Replace all numbers in "Hello 3 Im 2 years old" with NUMBERS',
                (
                    '{"fn_name": "fn_substitute_string_with_regex",'
                    ' "args": {"source_string": "Hello 3 Im 2 years old",'
                    ' "regex": "[0-9]+", "replacement": "NUMBERS"}}'
                ),
            ),
            (
                'Replace all vowels in "Programming is fun" with asterisks',
                (
                    '{"fn_name": "fn_substitute_string_with_regex",'
                    ' "args": {"source_string": "Programming is fun",'
                    ' "regex": "[aeiouAEIOU]", "replacement": "*"}}'
                ),
            ),
            (
                'Substitute "cat" with "dog" in'
                ' "The cat sat on the mat with another cat"',
                (
                    '{"fn_name": "fn_substitute_string_with_regex",'
                    ' "args": {"source_string":'
                    ' "The cat sat on the mat with another cat",'
                    ' "regex": "cat", "replacement": "dog"}}'
                ),
            ),
        ]
        ex_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in examples)
        func_list = "\n".join(func_lines)
        return f"Functions:\n{func_list}\n\nExamples:\n{ex_text}\n\nQ: "

    def _encode_static_prompt(self) -> List[int]:
        return self.encode(self._static_prompt)

    def _forward(
        self, ids: List[int], pkv: Any
    ) -> Tuple[npt.NDArray[np.float32], Any]:
        tensor = torch.tensor(
            [ids], device=self.model._device, dtype=torch.long
        )
        with torch.no_grad():
            out = self.model._model(
                input_ids=tensor, past_key_values=pkv, use_cache=True
            )
        logits: npt.NDArray[np.float32] = np.array(
            out.logits[0, -1].tolist(), dtype=np.float32
        )
        return logits, out.past_key_values

    def _inject(self, text: str,
                pkv: Any) -> Tuple[npt.NDArray[np.float32], Any]:
        ids = self.encode(text)
        return self._forward(ids, pkv)

    def _generate_until(
        self,
        stop: str,
        logits: npt.NDArray[np.float32],
        pkv: Any,
        max_tokens: int = 64,
        context_name: str = "",
        worker_id: int = 0,
    ) -> Tuple[str, npt.NDArray[np.float32], Any]:
        generated = ""
        for step in range(max_tokens):
            next_id = int(np.argmax(logits))
            token = self.vocab.get(next_id, "")
            if self.verbose:
                self._verbose_callback(
                    f"[worker {worker_id}][{context_name}] paso {step+1}: "
                    f"token='{token}' (id={next_id}) "
                    f" | acum='{generated}{token}'"
                )
            combined = generated + token
            if stop in combined:
                result = combined[: combined.index(stop)]
                if self.verbose:
                    self._verbose_callback(
                        f"[worker {worker_id}][{context_name}] -> '{result}'"
                    )
                return result, logits, pkv
            generated = combined
            logits, pkv = self._forward([next_id], pkv)
        if self.verbose:
            self._verbose_callback(
                f"[worker {worker_id}]"
                f"[{context_name} incompleto] -> '{generated}'"
            )
        return generated, logits, pkv

    def _generate_single(
        self,
        user_prompt: str,
        max_value_tokens: int = 64,
        use_cache: bool = True,
        worker_id: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        if use_cache and user_prompt in self._cache:
            if self.verbose:
                self._verbose_callback(
                    f"[worker {worker_id}] cache hit '{user_prompt}'"
                )
            return self._cache[user_prompt]

        suffix = user_prompt + f'\nA: {_JSON_PREFIX} '
        full_ids = self._prefix_ids + self.encode(suffix)
        logits, pkv = self._forward(full_ids, None)

        fn_raw, logits, pkv = self._generate_until(
            '"', logits, pkv, context_name="nombre_función",
            worker_id=worker_id
        )
        fn_name = fn_raw.strip()

        if fn_name not in self._fn_by_name:
            if self.verbose:
                self._verbose_callback(
                    f"[worker {worker_id}] recuperación nombre '{fn_name}' "
                    "no encontrado"
                )
            corrected = _closest(fn_name, list(self._fn_by_name.keys()))
            if corrected and corrected in self._fn_by_name:
                fn_name = corrected
            else:
                raise ValueError(f"Función no reconocida: '{fn_name}'")

        fn_def = self._fn_by_name[fn_name]
        param_keys = list(fn_def.parameters.keys())

        if not param_keys:
            result: Tuple[str, Dict[str, Any]] = (fn_name, {})
            if use_cache:
                self._update_cache(user_prompt, result)
            return result

        args: Dict[str, Any] = {}
        for i, key in enumerate(param_keys):
            param_type = fn_def.parameters[key].type
            is_last = i == len(param_keys) - 1
            if i == 0:
                scaffold = f'", "args": {{"{key}": '
            else:
                scaffold = f', "{key}": '
            logits, pkv = self._inject(scaffold, pkv)

            if param_type == "number":
                stop_char = "}" if is_last else ","
                value_str, logits, pkv = self._generate_until(
                    stop_char, logits, pkv, max_value_tokens,
                    f"arg {key} (num)", worker_id
                )
                clean = value_str.strip().rstrip("},").strip()
                try:
                    args[key] = float(clean)
                except ValueError:
                    recovered = _extract_number(clean)
                    if self.verbose:
                        self._verbose_callback(
                            f"[worker {worker_id}] "
                            f"parseo '{clean}' -> {recovered}"
                        )
                    args[key] = recovered
            else:
                logits, pkv = self._inject('"', pkv)
                value_str, logits, pkv = self._generate_until(
                    '"', logits, pkv, max_value_tokens,
                    f"arg {key} (str)", worker_id
                )
                args[key] = value_str

        result = (fn_name, args)
        if use_cache:
            self._update_cache(user_prompt, result)
        return result

    def generate(
        self,
        user_prompt: str,
        max_value_tokens: int = 64,
        use_cache: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generación guiada para un solo prompt."""
        return self._generate_single(user_prompt, max_value_tokens, use_cache)

    def generate_batch(
        self,
        prompts: List[str],
        max_value_tokens: int = 64,
        use_cache: bool = True,
        max_workers: int = 4,
        progress_callback: Optional[
            Callable[[int, str, Tuple[str, Dict[str, Any]], int], None]
        ] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        results: List[Optional[Tuple[str,
                                     Dict[str, Any]]]] = [None] * len(prompts)

        def _worker(
            idx: int, prompt: str, wid: int
        ) -> Tuple[int, int, Tuple[str, Dict[str, Any]]]:
            return idx, wid, self._generate_single(
                prompt, max_value_tokens, use_cache, wid
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            next_wid = 0
            for i, p in enumerate(prompts):
                wid = next_wid % max_workers
                next_wid += 1
                futures[executor.submit(_worker, i, p, wid)] = (i, wid)
            for future in as_completed(futures):
                idx, wid, result = future.result()
                results[idx] = result
                if progress_callback:
                    progress_callback(idx, prompts[idx], result, wid)

        return [r for r in results if r is not None]

    def _update_cache(self,
                      prompt: str, result: Tuple[str, Dict[str, Any]]) -> None:
        if (self._cache_size is not None and
                len(self._cache) >= self._cache_size):
            oldest = next(iter(self._cache.keys()))
            del self._cache[oldest]
        self._cache[prompt] = result


def _closest(name: str, candidates: List[str]) -> str:
    return max(candidates, key=lambda c: sum(a == b for a, b in zip(name, c)))


def _extract_number(text: str) -> float:
    buf = ""
    for ch in text:
        if ch.isdigit() or ch in (".", "-"):
            buf += ch
        elif buf:
            break
    try:
        return float(buf)
    except ValueError:
        return 0.0
