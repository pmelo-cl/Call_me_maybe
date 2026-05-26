"""Implementación optimizada para CPU con soporte regex y formato adecuado."""
import json
from typing import List, Dict, Tuple, Any

import numpy as np

from .schema_utils import FunctionDefinition


class ConstrainedDecoder:
    def __init__(self, model, vocab: Dict[int, str], functions: List[FunctionDefinition]):
        self.model = model
        self.vocab = vocab
        self.functions = functions
        self.function_names = [f.name for f in functions]

    def build_prompt(self, user_prompt: str) -> str:
        # Versión compacta de las funciones
        func_lines = []
        for f in self.functions:
            params = ', '.join(f.parameters.keys())
            func_lines.append(f"- {f.name}({params})")
        func_list = '\n'.join(func_lines)

        # Ejemplos que cubren todos los casos, con formato exacto
        examples = [
            ('What is the sum of 2 and 3?', 
             '{"fn_name": "fn_add_numbers", "args": {"a": 2.0, "b": 3.0}}'),
            ('Greet Maria', 
             '{"fn_name": "fn_greet", "args": {"name": "Maria"}}'),
            ('Reverse the string \'hello\'', 
             '{"fn_name": "fn_reverse_string", "args": {"s": "hello"}}'),
            ('What is the square root of 16?', 
             '{"fn_name": "fn_get_square_root", "args": {"a": 16.0}}'),
            ('Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS',
             '{"fn_name": "fn_substitute_string_with_regex", "args": {"source_string": "Hello 34 I\'m 233 years old", "regex": "[0-9]+", "replacement": "NUMBERS"}}'),
            ('Replace all vowels in \'Programming is fun\' with asterisks',
             '{"fn_name": "fn_substitute_string_with_regex", "args": {"source_string": "Programming is fun", "regex": "[aeiouAEIOU]", "replacement": "*"}}'),
            ('Substitute the word \'cat\' with \'dog\' in \'The cat sat on the mat with another cat\'',
             '{"fn_name": "fn_substitute_string_with_regex", "args": {"source_string": "The cat sat on the mat with another cat", "regex": "cat", "replacement": "dog"}}')
        ]
        ex_text = '\n'.join([f"Q: {q}\nA: {a}" for q, a in examples])

        prompt = f"""Functions:
{func_list}

Examples:
{ex_text}

Q: {user_prompt}
A: {{"fn_name":" """
        return prompt

    def generate(self, user_prompt: str, max_new_tokens: int = 40) -> Tuple[str, Dict[str, Any]]:
        full_prompt = self.build_prompt(user_prompt)
        input_ids = self.model.encode(full_prompt)[0].tolist()

        generated = ""
        for _ in range(max_new_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_id = int(np.argmax(logits))
            token = self.vocab.get(next_id, "")
            generated += token
            input_ids.append(next_id)
            if '}' in token and self._is_parseable('{"fn_name":"' + generated):
                break

        # Reconstruir JSON
        full_json = '{"fn_name":"' + generated
        start = full_json.find('{')
        end = full_json.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("No JSON")
        json_str = full_json[start:end+1]
        data = json.loads(json_str)

        fn_name = data["fn_name"].strip()
        args = data.get("args", {})

        # Convertir números a float si es necesario (para cumplir con el formato esperado)
        fn = next((f for f in self.functions if f.name == fn_name), None)
        if fn:
            for param, value in args.items():
                expected_type = fn.parameters[param].type
                if expected_type == "number" and isinstance(value, (int, float)):
                    args[param] = float(value)   # Asegurar que sea float (ej. 2.0)

        return fn_name, args

    def _is_parseable(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except:
            return False