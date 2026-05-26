"""Punto de entrada del programa."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel, Field

from .decoder import ConstrainedDecoder

try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    print("Error: No se encontró llm_sdk. Asegúrate de copiar la carpeta llm_sdk en la raíz del proyecto.")
    sys.exit(1)


class TestCase(BaseModel):
    prompt: str


class OutputRecord(BaseModel):
    prompt: str
    fn_name: str
    args: dict


class ParameterDef(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
    returns: Dict[str, str]


def load_function_definitions(path: Path) -> List[FunctionDefinition]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [FunctionDefinition(**item) for item in data]


def load_vocab(model) -> Dict[int, str]:
    """Carga el vocabulario y devuelve un diccionario id -> string real decodificado."""
    path = model.get_path_to_vocab_file()
    with open(path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)  # token_str -> id

    id_to_internal = {int(v): k for k, v in raw_vocab.items()}
    id_to_real_str = {}
    for token_id in id_to_internal.keys():
        decoded = model.decode([token_id])
        id_to_real_str[token_id] = decoded
    return id_to_real_str

def parse_arguments():
    parser = argparse.ArgumentParser(description="Function calling with constrained decoding.")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json",
                        help="Ruta al archivo JSON con los prompts de prueba.")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json",
                        help="Ruta donde se guardará el archivo JSON de resultados.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    base_dir = Path.cwd()
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    defs_path = base_dir / "data" / "input" / "functions_definition.json"

    if not input_path.exists():
        print(f"Error: Archivo de entrada no encontrado: {input_path}")
        sys.exit(1)
    if not defs_path.exists():
        print(f"Error: Archivo de definiciones de funciones no encontrado: {defs_path}")
        sys.exit(1)

    print("Loading function definitions...")
    functions = load_function_definitions(defs_path)
    print(f"Loaded {len(functions)} functions.")

    print(f"Loading test cases from '{input_path}'...")
    with open(input_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_cases = [TestCase(**item) for item in test_data]
    print(f"Loaded {len(test_cases)} test cases.")

    print("Initializing LLM model (Qwen/Qwen3-0.6B)...")
    model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

    print("Loading vocabulary...")
    id_to_str = load_vocab(model)
    print(f"Vocabulary size: {len(id_to_str)}")

    decoder = ConstrainedDecoder(model, id_to_str, functions)

    results: List[OutputRecord] = []
    total = len(test_cases)
    for i, case in enumerate(test_cases, 1):
        print(f"Processing prompt {i}/{total}: {case.prompt}")
        try:
            fn_name, args_dict = decoder.generate(case.prompt)
            results.append(OutputRecord(prompt=case.prompt, fn_name=fn_name, args=args_dict))
            print(f"  -> {fn_name}({args_dict})")
        except Exception as e:
            print(f"Error processing prompt '{case.prompt}': {e}")
            results.append(OutputRecord(prompt=case.prompt, fn_name="error", args={}))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to '{output_path}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, separators=(',', ': '))

    print("Done.")


if __name__ == "__main__":
    main()
