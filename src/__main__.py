"""Punto de entrada del programa."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel

from .decoder import ConstrainedDecoder
from .schema_utils import FunctionDefinition, load_function_definitions

try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    print(
        "Error: No se encontró llm_sdk. "
        "Asegúrate de copiar la carpeta llm_sdk en la raíz del proyecto."
    )
    sys.exit(1)

BAR_WIDTH = 40


def _bar(filled: int, total: int, width: int = BAR_WIDTH) -> str:
    """Devuelve una cadena de barra de progreso tipo [####----]."""
    done = int(width * filled / total) if total else width
    return f"[{'#' * done}{'-' * (width - done)}]"


def _fmt_seconds(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def progress(
    iterable: List[Any],
    desc: str,
    unit: str = "it",
) -> Iterator[Any]:
    """Itera sobre *iterable* mostrando una barra de progreso en línea."""
    total = len(iterable)
    t0 = time.monotonic()

    for i, item in enumerate(iterable):
        elapsed = time.monotonic() - t0
        rate = i / elapsed if elapsed > 0 else 0.0
        eta: Optional[float] = (total - i) / rate if rate > 0 else None
        eta_str = _fmt_seconds(eta) if eta is not None else "--:--"
        bar = _bar(i, total)
        line = (
            f"\r{desc}: {bar} {i}/{total} {unit} "
            f"[{_fmt_seconds(elapsed)}<{eta_str}]"
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        yield item

    elapsed = time.monotonic() - t0
    bar = _bar(total, total)
    sys.stdout.write(
        f"\r{desc}: {bar} {total}/{total} {unit} "
        f"[{_fmt_seconds(elapsed)}]\n"
    )
    sys.stdout.flush()


def spinner(desc: str) -> "SpinnerContext":
    """Devuelve un contexto que muestra una barra de progreso indeterminada."""
    return SpinnerContext(desc)


class SpinnerContext:
    """Context manager para tareas de duración desconocida."""

    _FRAMES = ["[>---]", "[--->]", "[--<-]", "[<---]"]

    def __init__(self, desc: str) -> None:
        self._desc = desc
        self._t0 = 0.0
        self._frame = 0

    def __enter__(self) -> "SpinnerContext":
        self._t0 = time.monotonic()
        sys.stdout.write(f"\r{self._desc}: {self._FRAMES[0]}  ")
        sys.stdout.flush()
        return self

    def tick(self) -> None:
        self._frame = (self._frame + 1) % len(self._FRAMES)
        elapsed = time.monotonic() - self._t0
        sys.stdout.write(
            f"\r{self._desc}: {self._FRAMES[self._frame]} "
            f"[{_fmt_seconds(elapsed)}]  "
        )
        sys.stdout.flush()

    def __exit__(self, *_: object) -> None:
        elapsed = time.monotonic() - self._t0
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        done = f"\r{self._desc}: [{'#' * BAR_WIDTH}] [{_fmt_seconds(elapsed)}]"
        sys.stdout.write(done.ljust(width) + "\n")
        sys.stdout.flush()


class TestCase(BaseModel):
    prompt: str


class OutputRecord(BaseModel):
    prompt: str
    fn_name: str
    args: Dict[str, Any]


def load_vocab(model: Any) -> Dict[int, str]:
    """Carga el vocabulario en O(V) usando convert_ids_to_tokens."""
    tokenizer = model._tokenizer
    vocab: Dict[str, int] = tokenizer.get_vocab()
    ids = list(vocab.values())
    # convert_ids_to_tokens es una operación de tabla hash,
    # ~150k entradas en <1s
    tokens: List[str] = tokenizer.convert_ids_to_tokens(ids)
    # Decodifica los bytes BPE (ej. "Ġhello" → " hello") en un solo batch
    decoded: List[str] = [
        tokenizer.convert_tokens_to_string([t]) for t in tokens
    ]
    return dict(zip(ids, decoded))


def load_test_cases(path: Path) -> List[TestCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TestCase(**item) for item in data]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Ruta al archivo JSON con los prompts de prueba.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Ruta donde se guardará el archivo JSON de resultados.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    base_dir = Path.cwd()
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    defs_path = base_dir / "data" / "input" / "functions_definition.json"

    for label, path in (("input", input_path), ("definitions", defs_path)):
        if not path.exists():
            print(f"Error: archivo de {label} no encontrado: {path}")
            sys.exit(1)

    functions: List[FunctionDefinition]
    with spinner("Loading definitions") as sp:
        functions = load_function_definitions(defs_path)
        sp.tick()
    print(f"  ✓ {len(functions)} functions loaded.")

    test_cases: List[TestCase]
    with spinner("Loading test cases") as sp:
        test_cases = load_test_cases(input_path)
        sp.tick()
    print(f"  ✓ {len(test_cases)} test cases loaded.")

    model: Any
    with spinner("Loading model  (Qwen3-0.6B)") as sp:
        model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
        sp.tick()

    with spinner("Loading vocab") as sp:
        id_to_str = load_vocab(model)
        sp.tick()
    print(f"    Vocabulary size: {len(id_to_str)}")

    decoder = ConstrainedDecoder(model, id_to_str, functions)
    with spinner("Warming up KV-cache") as sp:
        decoder._get_prefix_pkv()
        sp.tick()

    results: List[OutputRecord] = []
    for case in progress(test_cases, desc="Processing prompts", unit="prompt"):
        try:
            fn_name, args_dict = decoder.generate(case.prompt)
            results.append(
                OutputRecord(prompt=case.prompt, fn_name=fn_name,
                             args=args_dict)
            )
            print(f"    {fn_name}({args_dict})")
        except Exception as exc:  # noqa: BLE001
            print(f"    error en '{case.prompt}': {exc}")
            results.append(
                OutputRecord(prompt=case.prompt, fn_name="error", args={})
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with spinner("Writing results") as sp:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2,
                separators=(",", ": "),
            )
        sp.tick()
    print(f"    Done → '{output_path}'")


if __name__ == "__main__":
    main()
