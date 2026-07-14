*Este proyecto ha sido creado como parte del currículo de 42 por pmelo-cl.*

---

# call me maybe

## Descripción

**call me maybe** es una herramienta de *function calling* que traduce peticiones en lenguaje natural a llamadas de función estructuradas en JSON. Dado un prompt como `"What is the sum of 265 and 345?"`, el sistema no responde la pregunta — en su lugar produce el JSON exacto necesario para ejecutarla:

```json
{
  "prompt": "What is the sum of 265 and 345?",
  "fn_name": "fn_add_numbers",
  "args": { "a": 265.0, "b": 345.0 }
}
```

El reto central es la fiabilidad: los modelos pequeños (~0.6B parámetros) producen JSON válido menos del 30 % de las veces si se les deja generar libremente. Este proyecto alcanza el 100 % de JSON estructuralmente correcto mediante **generación guiada por fases**, una forma práctica de decodificación restringida que controla la salida del modelo token a token.

---

## Instrucciones

### Requisitos

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) — gestor de entornos y dependencias
- El paquete `llm_sdk` (copiar la carpeta `llm_sdk/` en la raíz del proyecto, al mismo nivel que `src/`)

### Instalación

```bash
uv sync
```

Crea el entorno virtual e instala todas las dependencias de `pyproject.toml`.

### Ejecución

```bash
# Rutas por defecto (data/input/ → data/output/)
uv run python -m src

# Rutas personalizadas
uv run python -m src --input data/input/my_tests.json --output data/output/results.json
```

### Flags disponibles

| Flag | Descripción |
|---|---|
| `--input <ruta>` | JSON de prompts de entrada (por defecto `data/input/function_calling_tests.json`) |
| `--output <ruta>` | JSON de resultados de salida (por defecto `data/output/function_calling_results.json`) |
| `--definitions <ruta>` | JSON de definiciones de funciones (por defecto `data/input/functions_definition.json`) |
| `--verbose` | Muestra la generación token a token en gris por debajo de los resultados |
| `--no-cache` | Desactiva la caché LRU de resultados |
| `--no-batch` | Procesa los prompts secuencialmente en lugar de en paralelo |
| `--workers N` | Número de workers para el modo batch (por defecto: automático según CPUs) |

### Otros comandos del Makefile

| Comando | Descripción |
|---|---|
| `make install` | Instala las dependencias con `uv` |
| `make run` | Ejecuta el programa con las rutas por defecto |
| `make debug` | Ejecuta con el depurador `pdb` de Python |
| `make clean` | Elimina `__pycache__`, `.mypy_cache` y similares |
| `make lint` | Ejecuta `flake8` + `mypy` con los flags obligatorios |
| `make lint-strict` | Ejecuta `flake8` + `mypy --strict` |

### Estructura del proyecto

```
.
├── src/
│   ├── __init__.py
│   ├── __main__.py        # CLI, carga de recursos, bucle de inferencia
│   ├── decoder.py         # Generación guiada (algoritmo central)
│   └── schema_utils.py    # Modelos Pydantic para las definiciones de función
├── llm_sdk/               # SDK del modelo (copiar aquí)
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output/            # Generado en ejecución, no incluido en el repo
├── pyproject.toml
├── uv.lock
└── Makefile
```

---

## Explicación del algoritmo

### El problema: LLMs y JSON libre

Un LLM genera texto token a token. En cada paso elige el token con mayor probabilidad (*greedy decoding*). Sin restricciones, el modelo puede producir JSON con comillas simples, llaves sin cerrar, campos inventados o texto de relleno — con una tasa de éxito de apenas el 30 %.

### Generación guiada por fases

En lugar de dejar al modelo generar el JSON completo y parsearlo, la generación se divide en fases donde **el modelo solo elige los valores** y el programa **inyecta la estructura fija**:

```
Prompt:  "Reverse the string 'world'"

Fase 1 — inyectamos:   Q: Reverse the string 'world'\nA: {"fn_name":"
           modelo →    fn_reverse_string          (genera hasta el '"' de cierre)

Fase 2 — inyectamos:   ", "args": {"s": "
           modelo →    world                      (genera hasta el '"' de cierre)

Resultado construido por el programa:
{"fn_name": "fn_reverse_string", "args": {"s": "world"}}
```

El JSON nunca puede estar malformado porque el programa construye la estructura; el modelo solo rellena valores en posiciones controladas.

### Flujo de `_generate_single()` paso a paso

```python
# 1. Construir el contexto completo: IDs del prefijo estático + IDs del sufijo dinámico
suffix   = user_prompt + '\nA: {"fn_name":" '
full_ids = self._prefix_ids + self.encode(suffix)

# 2. Un único forward pass sobre todo el contexto (past_key_values=None)
logits, pkv = self._forward(full_ids, None)

# 3. Fase 1 — el modelo elige el nombre de la función
fn_raw, logits, pkv = self._generate_until('"', logits, pkv)

# 4. Fase 2 — para cada parámetro de la función:
for key in param_keys:
    logits, pkv = self._inject(f'", "args": {{"{key}": ', pkv)  # estructura fija
    value, logits, pkv = self._generate_until(stop_char, logits, pkv)
    args[key] = float(value)   # o str, según el tipo declarado en la definición
```

### `_generate_until(stop, logits, pkv)` — generación token a token

```python
for step in range(max_tokens):
    next_id  = int(np.argmax(logits))      # token más probable
    token    = vocab[next_id]              # ID → string
    combined = generated + token
    if stop in combined:
        return combined[:combined.index(stop)], logits, pkv
    generated = combined
    logits, pkv = self._forward([next_id], pkv)   # un token, O(1) con KV-cache
```

La comprobación se hace sobre `combined` antes de añadir el token al buffer. Esto es necesario porque el tokenizador BPE puede agrupar varios caracteres en un solo token: `]"` puede ser un único token. Si el stop aparece dentro del token, se devuelve solo el prefijo anterior: `[aeiouAEIOU]` en lugar de `[aeiouAEIOU]"`.

### KV-cache incremental

Cada paso de `_generate_until` procesa **un solo token nuevo** apoyándose en el `past_key_values` acumulado:

```
Paso 0:   forward([prefix_ids + suffix_ids], pkv=None)  → pkv_N
Paso 1:   forward([token_1], pkv_N)                     → pkv_N+1
Paso 2:   forward([token_2], pkv_N+1)                   → pkv_N+2
```

Sin cache, producir el token k requiere reprocesar los k−1 anteriores (O(N²) total). Con cache, cada paso es O(1) en longitud de contexto.

### Modo batch con `ThreadPoolExecutor`

`generate_batch()` distribuye los prompts entre workers usando `concurrent.futures.ThreadPoolExecutor`. Cada worker llama a `_generate_single()` de forma independiente. Al completar cada prompt, se invoca un `progress_callback` que actualiza la barra y registra el resultado. El orden de los resultados se preserva por índice, no por orden de finalización.

### Caché LRU de resultados

`ConstrainedDecoder` mantiene un diccionario `_cache` de hasta `cache_size` entradas (por defecto 100). Si el mismo prompt se procesa dos veces, el resultado se devuelve inmediatamente sin ejecutar el modelo. Cuando el cache está lleno, se elimina la entrada más antigua (política FIFO sobre las claves del dict, que en Python 3.7+ mantienen orden de inserción). Se puede desactivar con `--no-cache`.

---

## Decisiones de diseño

**Generación guiada vs. enmascaramiento de logits**
El subject describe decodificación restringida clásica: en cada paso, los logits de los tokens inválidos se ponen a −∞. La generación guiada por fases es equivalente para este esquema concreto: en lugar de calcular qué tokens son válidos en cada paso, se inyectan directamente los tokens estructurales. El resultado es el mismo — JSON 100 % correcto — con menor complejidad de implementación.

**`pkv=None` en lugar de cachear el prefijo estático**
Transformers ≥ 4.38 devuelve un `DynamicCache` para Qwen3 — un objeto con estado interno complejo que no se puede clonar de forma segura. Intentarlo con `copy.copy()` o reconstrucción via `__class__()` producía errores en `get_seq_length()`. La solución es pasar `pkv=None` en cada llamada, reconstruyendo el cache desde los IDs del prefijo (ya tokenizados en `__init__` para no retokenizar en cada prompt).

**`verbose_callback` inyectable**
El decoder acepta un callable opcional `verbose_callback` para redirigir los mensajes de depuración. Esto desacopla la lógica de inferencia de la de presentación: en `__main__` se pasa una función que llama a `bar.log()` con color gris, pero el decoder no sabe nada de ANSI ni de barras de progreso.

**`ProgressBar` con hilo de refresco**
La barra se actualiza cada 0.2 s desde un hilo demonio independiente (`threading.Thread`), de modo que el tiempo transcurrido se muestra en tiempo real aunque no haya tokens generados en ese instante. Todas las escrituras a `stdout` están protegidas por un `threading.Lock` para evitar interleaving entre el hilo de refresco y el hilo principal.

**Silenciado de HuggingFace**
Los warnings de HF (`unauthenticated requests`, barra de carga de pesos) se suprimen redirigiendo `sys.stderr` a `/dev/null` durante la carga del modelo, y restaurándolo en el bloque `finally`. Esto garantiza que el terminal quede limpio sin depender de APIs privadas de HuggingFace que pueden cambiar entre versiones.

**Pydantic para todos los modelos de datos**
Requerimiento del subject. La validación ocurre en tiempo de carga: si `functions_definition.json` tiene un campo mal escrito, el error aparece con un mensaje claro antes de cargar el modelo.

---

## Análisis de rendimiento

| Métrica | Objetivo del subject | Resultado |
|---|---|---|
| JSON válido | 100 % | 100 % (por construcción) |
| Selección correcta de función | > 98 % | ~100 % en el conjunto de prueba |
| Tiempo total (11 prompts, CPU, `--no-batch`) | < 5 min | ~1 min |
| Tiempo total (11 prompts, CPU, batch automático) | < 3 min | ~40 s – 1 min |

Distribución aproximada del tiempo por fase:

```
Loading model:    ~4 s    (pesos en caché local tras la primera descarga)
Loading vocab:    <1 s    (convert_ids_to_tokens, una sola llamada)
Por prompt:       ~7 s    (forward pass completo + generación token a token)
```

El cuello de botella es el forward pass del modelo en CPU. En GPU el tiempo total bajaría a menos de 10 segundos para los 11 prompts.

---

## Retos encontrados

**Contaminación del KV-cache entre prompts**
El primer enfoque cacheaba el `past_key_values` del prefijo estático y lo clonaba para cada prompt. En Transformers ≥ 4.38, Qwen3 usa `DynamicCache` — un objeto cuyo estado interno no se copia de forma segura. El síntoma era que el segundo prompt heredaba el contexto del primero, produciendo salidas como `"s": "helloReverse the string 'hello'\nA: {"`. La solución fue pasar `pkv=None` y reconstruir el cache completo en cada llamada.

**Stop token dentro de un token BPE multi-carácter**
El tokenizador BPE puede agrupar varios caracteres en un solo token. `]"` puede ser un único token. La lógica original añadía el token al buffer y luego buscaba el stop, incluyendo la comilla en el valor. La solución es comprobar `combined = generated + token` antes de añadir al buffer, y devolver `combined[:combined.index(stop)]` si el stop aparece dentro del token.

**Interleaving entre hilo de refresco y hilo principal**
Con la barra de refresco automático en un hilo separado, sin un `Lock` las escrituras del hilo de refresco y las del hilo principal (resultado de un prompt) se intercalaban, produciendo líneas corruptas. Se protegieron todas las escrituras a `stdout` dentro de `ProgressBar` con `threading.Lock`.

**Warnings de HuggingFace en el terminal**
HuggingFace imprime una barra de progreso de carga de pesos y un warning de autenticación directamente en `stderr`, sin pasar por el sistema de logging de Python. Las variables de entorno (`TRANSFORMERS_VERBOSITY`, `HF_HUB_DISABLE_TELEMETRY`) no son suficientes para suprimirlos todos en todas las versiones. La solución robusta es redirigir `sys.stderr` a `/dev/null` durante la carga del modelo y restaurarlo en un bloque `finally`.

**Comillas simples en los prompts de entrada**
Los prompts del archivo de tests usan comillas simples para delimitar strings: `"Reverse the string 'world'"`. Con generación libre, la comilla simple del prompt se filtraba al valor del argumento, produciendo JSON inválido. Con la generación guiada esto deja de ser un problema: el modelo genera el valor entre comillas dobles inyectadas por el programa.

---

## Estrategia de pruebas

1. **Tests unitarios** — `_generate_until` con stop dentro de un token multi-carácter, `_extract_number` con entradas malformadas, `_closest` con nombres parcialmente correctos, `ProgressBar.log` con concurrencia simulada.

2. **Tests end-to-end** sobre `function_calling_tests.json` — verificando que cada registro del output parsea como JSON válido y que `fn_name` y `args` coinciden con los valores esperados.

3. **Casos límite probados:**
   - Prompts con comillas simples: `"Reverse the string 'world'"`
   - Números grandes: `"What is the sum of 265 and 345?"`
   - Regex con caracteres especiales: `"[aeiouAEIOU]"`, `"[0-9]+"`
   - Strings con apóstrofes: `"Hello 34 I'm 233 years old"`
   - Funciones con tres parámetros: `fn_substitute_string_with_regex`
   - Modo `--verbose` con batch y sin batch simultáneamente

4. **Verificación de linters** — `flake8` y `mypy` con los flags del subject pasan sin errores en todos los archivos de `src/`.

---

## Ejemplos de uso

### Ejecución básica

```bash
uv sync
uv run python -m src
```

Salida en terminal:

```
Loading definitions:     [####################################] [00:00]
  ✓ 5 function(s) loaded.
Loading test cases:      [####################################] [00:00]
  ✓ 11 test case(s) loaded.
Loading model:           [####################################] [00:04]
  ✓ Model loaded.
Loading vocab:           [####################################] [00:00]
  ✓ 151669 tokens in vocabulary.
  ✓ fn_add_numbers  {'a': 2.0, 'b': 3.0}
  ✓ fn_add_numbers  {'a': 265.0, 'b': 345.0}
  ✓ fn_greet  {'name': 'Shrek'}
  ...
Processing prompts:      [####################################]   11/11 [01:18]

Writing results:         [####################################] [00:00]
  ✓ Results written to 'data/output/function_calling_results.json'.
```

### Ejecución con verbose

```bash
uv run python -m src --verbose --no-batch
```

Los mensajes de depuración (token a token) aparecen en gris entre los resultados, sin interferir con la barra de progreso.

### Ejecución con batch explícito

```bash
uv run python -m src --workers 4
```

### Formato de entrada

`data/input/function_calling_tests.json`:
```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Reverse the string 'world'" }
]
```

`data/input/functions_definition.json`:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers",
    "parameters": { "a": {"type": "number"}, "b": {"type": "number"} },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string",
    "parameters": { "s": {"type": "string"} },
    "returns": {"type": "string"}
  }
]
```

### Formato de salida

`data/output/function_calling_results.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Reverse the string 'world'",
    "fn_name": "fn_reverse_string",
    "args": { "s": "world" }
  }
]
```

---

## Recursos

### Artículos y documentación

- Willard & Louf (2023) — *Efficient Guided Generation for Large Language Models* — paper fundacional sobre decodificación restringida basada en FSM. [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)
- Hokamp & Liu (2017) — *Lexically Constrained Decoding for Sequence Generation* — trabajo pionero sobre restricciones a nivel de token.
- [Outlines](https://github.com/dottxt-ai/outlines) — implementación open-source de referencia de decodificación restringida (uso de la librería prohibido en este proyecto, pero el código fuente es instructivo).
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — documentación del modelo base utilizado.
- [HuggingFace Transformers — KV Cache](https://huggingface.co/docs/transformers/kv_cache) — documentación sobre `DynamicCache` y `past_key_values`.
- [Python `concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) — documentación del `ThreadPoolExecutor` usado en el modo batch.
- [uv documentation](https://docs.astral.sh/uv/) — gestor de entornos y dependencias utilizado.

### Uso de IA

Se utilizó Claude (Anthropic) como herramienta de asistencia en las siguientes partes del proyecto:

- **Depuración de bugs**: diagnóstico de la contaminación del `DynamicCache` entre prompts, del bug del stop token dentro de tokens BPE multi-carácter, y del interleaving de escrituras entre hilos en `ProgressBar`.
- **Diseño del output en terminal**: implementación de la barra de progreso con hilo de refresco automático, `threading.Lock` para concurrencia segura, y silenciado de los warnings de HuggingFace.
- **Calidad del código**: type hints completos, docstrings, y pasar `flake8` + `mypy` con los flags del subject.

Todo el código generado con asistencia de IA fue revisado, comprendido y validado manualmente antes de su integración. La arquitectura central fue diseñada y razonada de forma propia.