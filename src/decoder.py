"""Generación guiada por fases: el modelo elige valores, nosotros construimos el JSON."""
from typing import Any, Dict, List, Tuple

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
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.functions = functions
        self._fn_by_name: Dict[str, FunctionDefinition] = {
            f.name: f for f in functions
        }
        self._static_prompt: str = self._build_static_prompt()
        self._prefix_ids: List[int] = self._encode_static_prompt()

    # ── Prompt ───────────────────────────────────────────────────────────────

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
                'Replace all numbers in "Hello 34 Im 233 years old" with NUMBERS',
                (
                    '{"fn_name": "fn_substitute_string_with_regex",'
                    ' "args": {"source_string": "Hello 34 Im 233 years old",'
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
        """Codifica el prompt estático una vez y devuelve sus IDs."""
        return self.model.encode(self._static_prompt)[0].tolist()

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _forward(
        self,
        ids: List[int],
        pkv: Any,
    ) -> Tuple[npt.NDArray[np.float32], Any]:
        """Forward pass sobre *ids* con KV-cache; devuelve logits y nuevo pkv."""
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

    def _inject(
        self,
        text: str,
        pkv: Any,
    ) -> Tuple[npt.NDArray[np.float32], Any]:
        """Inyecta *text* como tokens forzados; devuelve logits del siguiente token."""
        ids = self.model.encode(text)[0].tolist()
        return self._forward(ids, pkv)

    def _generate_until(
        self,
        stop: str,
        logits: npt.NDArray[np.float32],
        pkv: Any,
        max_tokens: int = 64,
    ) -> Tuple[str, npt.NDArray[np.float32], Any]:
        """
        Genera tokens hasta que *stop* aparezca en el texto acumulado.

        El stop se busca token a token ANTES de añadir cada nuevo token
        al buffer, para evitar que un token multi-carácter (p.ej. '"]')
        oculte el stop y se incluya en el valor generado.

        Devuelve el texto antes del stop, los últimos logits y el pkv.
        """
        generated = ""
        for _ in range(max_tokens):
            next_id = int(np.argmax(logits))
            token = self.vocab.get(next_id, "")

            # Si el stop está dentro del token, añadimos solo lo previo al stop
            combined = generated + token
            if stop in combined:
                return combined[: combined.index(stop)], logits, pkv

            generated = combined
            logits, pkv = self._forward([next_id], pkv)

        return generated, logits, pkv

    # ── Guided generation ─────────────────────────────────────────────────────

    def generate(
        self,
        user_prompt: str,
        max_value_tokens: int = 64,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generación guiada por fases.

        Reconstruye el contexto completo desde el prefijo estático en cada llamada,
        evitando problemas de mutabilidad del KV-cache entre prompts distintos.
        """
        # 1. Construir la secuencia completa de IDs del prefijo + usuario + inicio JSON
        suffix = user_prompt + f'\nA: {_JSON_PREFIX} '
        full_ids = self._prefix_ids + self.model.encode(suffix)[0].tolist()

        # 2. Forward inicial sin KV-cache (past_key_values=None)
        logits, pkv = self._forward(full_ids, None)

        # ── Fase 1: nombre de la función ──────────────────────────────────────
        fn_raw, logits, pkv = self._generate_until('"', logits, pkv)
        fn_name = fn_raw.strip()

        if fn_name not in self._fn_by_name:
            fn_name = _closest(fn_name, list(self._fn_by_name.keys()))

        fn_def = self._fn_by_name[fn_name]
        param_keys = list(fn_def.parameters.keys())

        if not param_keys:
            return fn_name, {}

        # ── Fase 2: valores de los argumentos ─────────────────────────────────
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
                    stop_char, logits, pkv, max_tokens=max_value_tokens
                )
                clean = value_str.strip().rstrip("},").strip()
                try:
                    args[key] = float(clean)
                except ValueError:
                    args[key] = _extract_number(clean)
            else:
                # Strings: inyectamos '"' de apertura; el modelo genera hasta '"'
                logits, pkv = self._inject('"', pkv)
                value_str, logits, pkv = self._generate_until(
                    '"', logits, pkv, max_tokens=max_value_tokens
                )
                args[key] = value_str

        return fn_name, args


# ── Module helpers ────────────────────────────────────────────────────────────


def _closest(name: str, candidates: List[str]) -> str:
    """Candidato con mayor solapamiento de prefijo con *name*."""
    return max(candidates, key=lambda c: sum(a == b for a, b in zip(name, c)))


def _extract_number(text: str) -> float:
    """Extrae el primer float/int de *text*; devuelve 0.0 si no hay ninguno."""
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