"""Punto de entrada del programa."""
import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError


from .decoder import ConstrainedDecoder
from .schema_utils import (
                            FunctionDefinition, TestCase,
                            OutputRecord, load_function_definitions
                          )

try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    print(
        "Error: No se encontró llm_sdk. "
        "Asegúrate de copiar la carpeta llm_sdk en la raíz del proyecto."
    )
    sys.exit(1)

GREEN = "\033[92m"
GRAY = "\033[90m"
RESET = "\033[0m"
_ERASE_LINE = "\033[2K"

_FULL_BAR_WIDTH = 36
_LABEL_W = 24


def _bar(filled: int, total: int, width: int = _FULL_BAR_WIDTH) -> str:
    done = int(width * filled / total) if total else width
    return f"[{'#' * done}{'·' * (width - done)}]"


def _fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def _bar_line(desc: str, filled: int, total: int, elapsed: float) -> str:
    rate = filled / elapsed if elapsed > 0 else 0.0
    eta: Optional[float] = (
     (total - filled) / rate
     if rate > 0 and filled
     else None
    )
    eta_str = _fmt(eta) if eta is not None else "--:--"
    bar = _bar(filled, total)
    label = f"{desc}:".ljust(_LABEL_W)
    return f"{label} {bar} {filled:>4}/{total} [{_fmt(elapsed)}<{eta_str}]"


class ProgressBar:
    """Barra de progreso con actualización automática del tiempo."""

    def __init__(self, desc: str, total: int,
                 refresh_interval: float = 0.2) -> None:
        self.desc = desc
        self.total = total
        self.filled = 0
        self.t0 = time.monotonic()
        self._lock = threading.Lock()
        self._running = True
        self._refresh_interval = refresh_interval
        self._redraw()
        self._timer_thread = threading.Thread(target=self._auto_refresh,
                                              daemon=True)
        self._timer_thread.start()

    def _auto_refresh(self) -> None:
        while self._running:
            time.sleep(self._refresh_interval)
            if self._running:
                with self._lock:
                    self._redraw()

    def _redraw(self) -> None:
        elapsed = time.monotonic() - self.t0
        line = _bar_line(self.desc, self.filled, self.total, elapsed)
        sys.stdout.write(f"\r{_ERASE_LINE}{line}")
        sys.stdout.flush()

    def log(self, msg: str) -> None:
        """Imprime un mensaje encima de la barra y redibuja la barra."""
        with self._lock:
            sys.stdout.write("\r")
            sys.stdout.write(_ERASE_LINE)
            sys.stdout.write("\n")
            print(msg)
            self._redraw()

    def advance(self, n: int = 1) -> None:
        with self._lock:
            self.filled = min(self.filled + n, self.total)
            self._redraw()

    def set(self, filled: int) -> None:
        with self._lock:
            self.filled = min(filled, self.total)
            self._redraw()

    def finish(self) -> None:
        """Detiene el hilo de refresco y muestra la barra final."""
        self._running = False
        if hasattr(self, '_timer_thread'):
            self._timer_thread.join(timeout=0.5)
        elapsed = time.monotonic() - self.t0
        label = f"{self.desc}:".ljust(_LABEL_W)
        bar = _bar(self.total, self.total)
        sys.stdout.write(
            f"\r{_ERASE_LINE}{label} {bar} {self.total:>4}/{self.total} "
            f"[{_fmt(elapsed)}]\n"
        )
        sys.stdout.flush()


class SimpleTask:
    """Context manager para tareas de duración desconocida."""

    def __init__(self, desc: str) -> None:
        self.desc = desc
        self.t0 = 0.0

    def __enter__(self) -> "SimpleTask":
        self.t0 = time.monotonic()
        label = f"{self.desc}:".ljust(_LABEL_W)
        sys.stdout.write(f"{label} ...")
        sys.stdout.flush()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed = time.monotonic() - self.t0
        sys.stdout.write("\r")
        label = f"{self.desc}:".ljust(_LABEL_W)
        bar = f"[{'#' * _FULL_BAR_WIDTH}]"
        sys.stdout.write(f"{label} {bar} [{_fmt(elapsed)}]\n")
        sys.stdout.flush()


def task(desc: str) -> SimpleTask:
    return SimpleTask(desc)


def load_vocab(model: Any) -> Dict[int, str]:
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


def compute_workers(num_prompts: int,
                    user_workers: Optional[int] = None) -> int:
    if user_workers is not None:
        return max(1, user_workers)
    cpu_count = os.cpu_count() or 4
    workers = max(1, min(cpu_count, num_prompts // 2))
    return min(workers, num_prompts)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    parser.add_argument(
        "--input", type=str, default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/output/function_calling_results.json"
    )
    parser.add_argument(
        "--definitions", type=str,
        default="data/input/functions_definition.json",
        help="Ruta al archivo JSON con las definiciones de funciones"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Muestra generación token a token"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Desactiva caché de resultados"
    )
    parser.add_argument(
        "--no-batch", action="store_true", help="Procesa secuencialmente"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Número de workers"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path.cwd()
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    defs_path = base_dir / args.definitions
    for label, path in (("input", input_path), ("definitions", defs_path)):
        if not path.exists():
            sys.exit(f"Error: archivo de {label} no encontrado: {path}")

    try:
        with task("Loading definitions"):
            functions: List[FunctionDefinition]
            functions = load_function_definitions(defs_path)
    except json.JSONDecodeError as e:
        sys.exit("Error: el archivo de definiciones no "
                 f"es un JSON válido: {e}")
    except ValidationError as e:
        sys.exit("Error: el contenido del JSON no cumple el"
                 f"esquema esperado:\n{e}")
    except Exception as e:
        sys.exit(f"Error inesperado al cargar definiciones: {e}")
    print(f"  ✓ {len(functions)} function(s) loaded.")

    try:
        with task("Loading test cases"):
            test_cases: List[TestCase] = load_test_cases(input_path)
    except json.JSONDecodeError as e:
        sys.exit("Error: el archivo de tests no "
                 f"es un JSON válido: {e}")
    except ValidationError as e:
        sys.exit("Error: el contenido del JSON no cumple el"
                 f"esquema esperado:\n{e}")
    except Exception as e:
        sys.exit(f"Error inesperado al cargar tests: {e}")
    print(f"  ✓ {len(test_cases)} test case(s) loaded.")

    devnull = open(os.devnull, "w")
    original_stderr = sys.stderr
    sys.stderr = devnull
    try:
        with task("Loading model"):
            model: Any = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    finally:
        sys.stderr = original_stderr
        devnull.close()
    print("  ✓ Model loaded.")

    with task("Loading vocab"):
        id_to_str = load_vocab(model)
    print(f"  ✓ {len(id_to_str)} tokens in vocabulary.")

    bar = ProgressBar("Processing prompts", len(test_cases))

    def verbose_callback(msg: str) -> None:
        escaped = msg.replace('\n', '\\n').replace('\r',
                                                   '\\r').replace('\t', '\\t')
        bar.log(f"{GRAY}{escaped}{RESET}")

    decoder = ConstrainedDecoder(
        model,
        id_to_str,
        functions,
        verbose=args.verbose,
        verbose_callback=verbose_callback if args.verbose else None,
        cache_size=None if args.no_cache else 100,
    )

    prompts = [case.prompt for case in test_cases]
    total = len(prompts)
    results: List[OutputRecord] = []

    if args.no_batch:
        for idx, prompt in enumerate(prompts):
            try:
                fn_name, args_dict = decoder.generate(
                    prompt, use_cache=not args.no_cache
                )
                results.append(
                    OutputRecord(prompt=prompt,
                                 fn_name=fn_name, args=args_dict)
                )
                bar.log(f"{GREEN}  ✓ {fn_name}  {args_dict}{RESET}")
            except Exception as exc:
                results.append(
                    OutputRecord(prompt=prompt, fn_name="error", args={})
                )
                bar.log(f"  ✗ {prompt!r}: {exc}")
            bar.set(idx + 1)
    else:
        workers = compute_workers(total, args.workers)
        bar.log(f"  → batching: {workers} worker(s) para {total} prompt(s).")
        results_holder: List[Optional[OutputRecord]] = [None] * total

        def on_progress(
            idx: int,
            prompt: str,
            result: Tuple[str, Dict[str, Any]],
            wid: int
        ) -> None:
            fn_name, args_dict = result
            results_holder[idx] = OutputRecord(
                prompt=prompt, fn_name=fn_name, args=args_dict
            )
            completed = sum(1 for r in results_holder if r is not None)
            wlabel = f"[w{wid}] " if not args.no_batch else ""
            bar.log(f"{GREEN}  ✓ {wlabel}{fn_name}  {args_dict}{RESET}")
            bar.set(completed)

        decoder.generate_batch(
            prompts,
            use_cache=not args.no_cache,
            max_workers=workers,
            progress_callback=on_progress,
        )
        results = [r for r in results_holder if r is not None]

    bar.finish()
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with task("Writing results"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2,
                separators=(",", ": "),
            )
    print(f"  ✓ Results written to '{output_path}'.")


if __name__ == "__main__":
    main()
