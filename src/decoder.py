import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from .schema_utils import FunctionDefinition

# Prefijo fijo que el decoder añade antes del JSON generado
_JSON_PREFIX = '{"fn_name":"'

# Número de llaves abiertas que necesita el JSON de salida: { args: { } }
_EXPECTED_DEPTH = 2


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

        self._static_prompt: str = self._build_static_prompt()
        self._prefix_ids: List[int] = []
        self._prefix_pkv: Optional[Any] = None

    def _build_static_prompt(self) -> str:
        """Parte del prompt que no cambia entre llamadas."""
        func_lines = [
            f"- {f.name}({', '.join(f.parameters.keys())})"
            for f in self.functions
        ]
        func_list = "\n".join(func_lines)

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
                'Substitute the word "cat" with "dog" in'
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
        return f"Functions:\n{func_list}\n\nExamples:\n{ex_text}\n\nQ: "

    def _get_prefix_pkv(self) -> Tuple[List[int], Any]:
        """Procesa el prompt estático una vez y guarda los past_key_values."""
        if self._prefix_pkv is not None:
            return self._prefix_ids, self._prefix_pkv

        ids = self.model.encode(self._static_prompt)[0].tolist()
        tensor = torch.tensor([ids], device=self.model._device,
                              dtype=torch.long)
        with torch.no_grad():
            out = self.model._model(input_ids=tensor, use_cache=True)
        self._prefix_ids = ids
        self._prefix_pkv = out.past_key_values
        return ids, self._prefix_pkv

    def generate(
        self,
        user_prompt: str,
        max_new_tokens: int = 128,
    ) -> Tuple[str, Dict[str, Any]]:
        """Genera fn_name y args usando KV-cache para el prefijo estático."""
        _, pkv = self._get_prefix_pkv()

        # Sanitiza el user_prompt: reemplaza comillas simples por dobles
        # para que no rompan el JSON generado
        clean_prompt = user_prompt.replace("'", '"')

        suffix_text = clean_prompt + f'\nA: {_JSON_PREFIX} '
        suffix_ids: List[int] = self.model.encode(suffix_text)[0].tolist()

        tensor = torch.tensor(
            [suffix_ids], device=self.model._device, dtype=torch.long
        )
        with torch.no_grad():
            out = self.model._model(
                input_ids=tensor,
                past_key_values=pkv,
                use_cache=True,
            )
        logits_np: npt.NDArray[np.float32] = np.array(
            out.logits[0, -1].tolist(), dtype=np.float32
        )
        past = out.past_key_values

        generated = ""
        next_id = int(np.argmax(logits_np))

        for _ in range(max_new_tokens):
            token = self.vocab.get(next_id, "")
            generated += token

            if self._is_complete(_JSON_PREFIX + generated):
                break

            tok_tensor = torch.tensor(
                [[next_id]], device=self.model._device, dtype=torch.long
            )
            with torch.no_grad():
                out = self.model._model(
                    input_ids=tok_tensor,
                    past_key_values=past,
                    use_cache=True,
                )
            logits_np = np.array(
                out.logits[0, -1].tolist(), dtype=np.float32
            )
            past = out.past_key_values
            next_id = int(np.argmax(logits_np))

        raw = _JSON_PREFIX + generated
        data = self._extract_json(raw)

        fn_name: str = data["fn_name"].strip()
        args: Dict[str, Any] = data.get("args", {})

        fn: Optional[FunctionDefinition] = next(
            (f for f in self.functions if f.name == fn_name), None
        )
        if fn:
            for param, value in args.items():
                expected_type = fn.parameters[param].type
                if expected_type == "number" and isinstance(value,
                                                            (int, float)):
                    args[param] = float(value)

        return fn_name, args

    @staticmethod
    def _is_complete(text: str) -> bool:
        """Devuelve True si *text* es un JSON completo y válido."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Extrae el primer objeto JSON de *text*.
        Si el JSON está truncado (falta uno o más '}'), intenta cerrarlo.
        """
        start = text.find("{")
        if start == -1:
            raise ValueError("No se encontró '{' en la salida generada.")

        fragment = text[start:]

        # Intento directo
        try:
            return json.loads(fragment)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Cierra llaves abiertas que falten (hasta 3 niveles de profundidad)
        depth = _count_open_braces(fragment)
        for closing in range(1, depth + 1):
            candidate = fragment + "}" * closing
            try:
                return json.loads(candidate)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                continue

        # Último recurso: extrae hasta el último '}' presente
        end = text.rfind("}")
        if end == -1:
            raise ValueError("No se encontró '}' en la salida generada.")
        return json.loads(text[start: end + 1])  # type: ignore[no-any-return]


def _count_open_braces(text: str) -> int:
    """Cuenta cuántas llaves '{' están sin cerrar en text"""
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    return max(depth, 0)
