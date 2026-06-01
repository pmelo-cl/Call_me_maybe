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

BAR_WIDTH = 36
_LABEL_W = 24   # ancho fijo para las etiquetas → columnas alineadas


def _bar(filled: int, total: int, width: int = BAR_WIDTH) -> str:
    done = int(width * filled / total) if total else width
    return f"[{'#' * done}{'·' * (width - done)}]"


def _fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def _term_width() -> int:
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def _clear_line() -> None:
    """Sobreescribe la línea actual con espacios y vuelve al inicio."""
    sys.stdout.write(f"\r{' ' * _term_width()}\r")
    sys.stdout.flush()


def _print_bar(desc: str, filled: int, total: int, elapsed: float) -> None:
    """Escribe la barra de progreso en la línea actual (sin newline)."""
    rate = filled / elapsed if elapsed > 0 else 0.0
    eta: Optional[float] = (total - filled) / rate if rate > 0 and filled else None
    eta_str = _fmt(eta) if eta is not None else "--:--"
    bar = _bar(filled, total)
    label = f"{desc}:".ljust(_LABEL_W)
    line = f"\r{label} {bar} {filled:>4}/{total} [{_fmt(elapsed)}<{eta_str}]"
    sys.stdout.write(line)
    sys.stdout.flush()


def log(msg: str) -> None:
    """
    Imprime *msg* en su propia línea limpia.
    Debe usarse en lugar de print() dentro de bucles progress().
    Borra la barra activa, imprime el mensaje y la barra reaparecerá
    en la próxima iteración.
    """
    _clear_line()
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def progress(
    iterable: List[Any],
    desc: str,
    unit: str = "it",
) -> Iterator[Any]:
    """
    Itera sobre *iterable* mostrando una barra de progreso.

    La barra se dibuja ANTES de ceder cada elemento para que siempre
    esté visible mientras el trabajo ocurre. Cuando el bucle exterior
    necesite imprimir algo, debe llamar a log() en lugar de print()
    para no solaparse con la barra.
    """
    total = len(iterable)
    t0 = time.monotonic()

    for i, item in enumerate(iterable):
        _print_bar(desc, i, total, time.monotonic() - t0)
        yield item

    # Barra final al 100 %
    elapsed = time.monotonic() - t0
    label = f"{desc}:".ljust(_LABEL_W)
    bar = _bar(total, total)
    sys.stdout.write(
        f"\r{label} {bar} {total:>4}/{total} [{_fmt(elapsed)}]\n"
    )
    sys.stdout.flush()


class SpinnerContext:
    """
    Context manager para tareas de duración desconocida.
    Al salir imprime la barra llena con el tiempo total.
    """

    _FRAMES = ["[>···]", "[·>··]", "[··>·]", "[···>]"]

    def __init__(self, desc: str) -> None:
        self._desc = desc
        self._t0 = 0.0
        self._frame = 0

    def __enter__(self) -> "SpinnerContext":
        self._t0 = time.monotonic()
        label = f"{self._desc}:".ljust(_LABEL_W)
        sys.stdout.write(f"\r{label} {self._FRAMES[0]}")
        sys.stdout.flush()
        return self

    def tick(self) -> None:
        self._frame = (self._frame + 1) % len(self._FRAMES)
        elapsed = time.monotonic() - self._t0
        label = f"{self._desc}:".ljust(_LABEL_W)
        sys.stdout.write(
            f"\r{label} {self._FRAMES[self._frame]} [{_fmt(elapsed)}]  "
        )
        sys.stdout.flush()

    def __exit__(self, *_: object) -> None:
        elapsed = time.monotonic() - self._t0
        label = f"{self._desc}:".ljust(_LABEL_W)
        bar = f"[{'#' * BAR_WIDTH}]"
        line = f"\r{label} {bar} [{_fmt(elapsed)}]"
        sys.stdout.write(line.ljust(_term_width()) + "\n")
        sys.stdout.flush()


def spinner(desc: str) -> SpinnerContext:
    return SpinnerContext(desc)


class TestCase(BaseModel):
    prompt: str


class OutputRecord(BaseModel):
    prompt: str
    fn_name: str
    args: Dict[str, Any]


def load_vocab(model: Any) -> Dict[int, str]:
    """Carga el vocabulario en O(V) sin forward pass usando el tokenizer."""
    tokenizer = model._tokenizer
    vocab: Dict[str, int] = tokenizer.get_vocab()
    ids = list(vocab.values())
    tokens: List[str] = tokenizer.convert_ids_to_tokens(ids)
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
            sys.exit(f"Error: archivo de {label} no encontrado: {path}")

    functions: List[FunctionDefinition]
    with spinner("Loading definitions") as sp:
        functions = load_function_definitions(defs_path)
        sp.tick()
    print(f"  ✓ {len(functions)} function(s) loaded.")

    test_cases: List[TestCase]
    with spinner("Loading test cases") as sp:
        test_cases = load_test_cases(input_path)
        sp.tick()
    print(f"  ✓ {len(test_cases)} test case(s) loaded.")

    model: Any
    with spinner("Loading model") as sp:
        model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
        sp.tick()

    with spinner("Loading vocab") as sp:
        id_to_str = load_vocab(model)
        sp.tick()
    print(f"  ✓ {len(id_to_str)} tokens in vocabulary.")

    decoder = ConstrainedDecoder(model, id_to_str, functions)

    # NOTA: Ya no es necesario el "Warming up KV-cache" porque el decoder
    # reconstruye el contexto desde cero en cada llamada a generate().

    results: List[OutputRecord] = []
    for case in progress(test_cases, desc="Processing prompts", unit="prompt"):
        try:
            fn_name, args_dict = decoder.generate(case.prompt)
            results.append(
                OutputRecord(
                    prompt=case.prompt, fn_name=fn_name, args=args_dict
                )
            )
            log(f"  ✓ {fn_name}  {args_dict}")
        except Exception as exc:  # noqa: BLE001
            results.append(
                OutputRecord(prompt=case.prompt, fn_name="error", args={})
            )
            log(f"  ✗ {case.prompt!r}: {exc}")

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
    print(f"  ✓ Results written to '{output_path}'.")


if __name__ == "__main__":
    main()