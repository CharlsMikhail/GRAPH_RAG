# =============================================================================
# RAG GRAFO — GraphRAG v5.1 | Semáforo + Evaluación Secuencial + ctx Safe
# =============================================================================
#
# FIX v5.1 (sobre v5.0):
#   [1] Paso 4: asyncio.gather → for-loop secuencial (evita saturación VRAM)
#   [2] Contexto truncado a 4000 chars antes de enviar al LLM (protege num_ctx)
#   [3] Llaves JSON escapadas en prompts para evitar KeyError con .format()
#
# OLLAMA ENV (configurar fuera de Python antes de lanzar):
#   OLLAMA_NUM_PARALLEL=8
#   OLLAMA_MAX_LOADED_MODELS=2
# =============================================================================

import nest_asyncio
nest_asyncio.apply()

import asyncio
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import boto3
import fitz  # PyMuPDF
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Esto se ejecuta solo una vez al encender el servidor
    logger.info("🔥 Calentando motores: Precargando modelos en la VRAM de tu RTX 5070 Ti...")
    try:
        # Petición 'dummy' para despertar a Qwen
        await extractor_llm.acomplete("Despierta")
        logger.info("✅ Extractor precargado.")
        
        # Petición 'dummy' para despertar a Gemma
        await query_llm.acomplete("Despierta")
        logger.info("✅ Evaluador precargado.")
    except Exception as e:
        logger.warning(f"⚠️ Fallo al precargar modelos: {e}")
    
    yield  # Aquí es donde el servidor empieza a recibir tráfico real
    
    logger.info("🛑 Apagando servidor...")


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
OLLAMA_BASE_URL     = "http://localhost:11434"
MINIO_ENDPOINT      = "http://localhost:9000"
MINIO_ACCESS_KEY    = "admin"
MINIO_SECRET_KEY    = "password123"
BUCKET_NAME         = "test"
NEO4J_URL           = "bolt://localhost:7687"
NEO4J_USER          = "neo4j"
NEO4J_PASS          = "password123"

EXTRACTOR_LLM_MODEL = "qwen2.5:1.5b"
QUERY_LLM_MODEL     = "gemma3:latest"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
RERANKER_MODEL      = "BAAI/bge-reranker-base"

MAX_WORKERS         = 8
RERANKER_TOP_N      = 2
RETRIEVAL_TOP_K     = 3
MIN_CONTEXT_LENGTH  = 150   # chars mínimos para considerar contexto útil
MAX_CONTEXT_LENGTH  = 4000  # [FIX 2] Techo duro: protege num_ctx=4096 del query_llm

# Semáforo Nivel 1: vídeos procesándose en paralelo.
VIDEO_CONCURRENCY   = 4

ETIQUETAS_VALIDAS = {
    "CORRECTO",
    "MAYORMENTE_CORRECTO",
    "PARCIAL",
    "MAYORMENTE_INCORRECTO",
    "INCORRECTO",
    "SIN_EVIDENCIA",
}

SCORE_MAP = {
    "CORRECTO"             : 1.00,
    "MAYORMENTE_CORRECTO"  : 0.75,
    "PARCIAL"              : 0.50,
    "SIN_EVIDENCIA"        : 0.50,
    "MAYORMENTE_INCORRECTO": 0.25,
    "INCORRECTO"           : 0.00,
}

# =============================================================================
# APP + ESTADO GLOBAL
# =============================================================================
app = FastAPI(title="RAG Grafo", version="5.2", lifespan=lifespan)

graph_index   : Optional[PropertyGraphIndex] = None
chat_sessions : dict[str, object] = {}
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_video_semaphore = asyncio.Semaphore(VIDEO_CONCURRENCY)

# =============================================================================
# DETECCIÓN DE GPU
# =============================================================================
try:
    import torch
    _EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
except ImportError:
    _EMBED_DEVICE = "cpu"

# =============================================================================
# EMBEDDINGS
# =============================================================================
logger.info("Verificando modelo de embeddings...")
_st_model = SentenceTransformer(DEFAULT_EMBED_MODEL, device=_EMBED_DEVICE)
logger.info("✅ Embeddings listos.")

_hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=DEFAULT_EMBED_MODEL,
    device=_EMBED_DEVICE,
    cache_folder=_hf_cache,
)

# =============================================================================
# LLMs
# =============================================================================
extractor_llm = Ollama(
    model=EXTRACTOR_LLM_MODEL,
    temperature=0.0,
    request_timeout=300.0,
    base_url=OLLAMA_BASE_URL,
    additional_kwargs={
        "num_ctx"    : 8192,   # Soporta transcripciones de ~13.5k chars
        "num_predict": 1024,
    },
)

query_llm = Ollama(
    model=QUERY_LLM_MODEL,
    temperature=0.0,
    request_timeout=60.0,
    base_url=OLLAMA_BASE_URL,
    additional_kwargs={
        "num_ctx"    : 4096,   # Evaluación individual: prompts cortos
        "num_predict": 512,
        "keep_alive": -1
    },
)

# =============================================================================
# RERANKER
# =============================================================================
logger.info(f"Cargando reranker ({RERANKER_MODEL}) en {_EMBED_DEVICE}...")
reranker = SentenceTransformerRerank(
    model=RERANKER_MODEL,
    top_n=RERANKER_TOP_N,
    device=_EMBED_DEVICE,
)
logger.info("✅ Reranker listo.")

logger.info(
    f"✅ Setup v5.1 | Device: {_EMBED_DEVICE} | "
    f"Extractor: {EXTRACTOR_LLM_MODEL} (ctx 8192) | "
    f"Query: {QUERY_LLM_MODEL} (ctx 4096, max ctx enviado: {MAX_CONTEXT_LENGTH}) | "
    f"Semáforo: {VIDEO_CONCURRENCY} vídeos | top_k={RETRIEVAL_TOP_K}→top_n={RERANKER_TOP_N}"
)

# =============================================================================
# PROMPTS DEL SISTEMA
#
# [FIX 3] TODAS las llaves JSON dentro de los strings están escapadas (duplicadas)
#         para evitar KeyError al llamar a .format().
#         Las únicas llaves simples {placeholders} son las variables del formato.
# =============================================================================

PROMPT_EXTRACCION = """\
Eres un extractor de conceptos educativos MUY CONSERVADOR.
Extrae ÚNICAMENTE los conceptos que fueron explícitamente enseñados y explicados.

══════════════════════════════════════════════════
REGLAS OBLIGATORIAS
══════════════════════════════════════════════════
✅ INCLUIR solo si el concepto fue EXPLICADO directamente (no solo mencionado).
❌ NO INCLUIR si:
   - Solo fue mencionado de pasada sin explicación.
   - Es una inferencia o tema implícito no cubierto.
   - Es el entorno o herramienta (VSCode, terminal) salvo que sea el tema.
   - Es un ejemplo usado para ilustrar otro concepto.
   - Es un concepto avanzado no enseñado explícitamente.

⚠️  PRINCIPIO FUNDAMENTAL: Menos conceptos reales > más conceptos inventados.
    Si tienes duda, NO lo incluyas.

══════════════════════════════════════════════════
EJEMPLOS FEW-SHOT
══════════════════════════════════════════════════

Transcripción: "Una variable es un espacio en memoria que almacena un valor. \
Por ejemplo: edad = 25."
Respuesta correcta:
{{"conceptos": [{{"termino": "Variable", "definicion_video": "Espacio en memoria que almacena un valor."}}]}}

Respuesta INCORRECTA (inventar conceptos no enseñados):
{{"conceptos": [{{"termino": "Variable", "definicion_video": "explicacion_exacta_extraida_del_video"}}, {{"termino": "Memoria RAM", "definicion_video": "explicacion_exacta_extraida_del_video"}}, {{"termino": "Python", "definicion_video": "explicacion_exacta_extraida_del_video"}}]}}

---

Transcripción: "int almacena enteros como 1 o 100. float almacena decimales como 3.14."
Respuesta correcta:
{{"conceptos": [{{"termino": "int", "definicion_video": "Almacena números enteros como 1 o 100."}}, {{"termino": "float", "definicion_video": "Almacena decimales como 3.14."}}]}}

---

Transcripción: "Ahora abrimos Visual Studio Code y escribimos nuestro primer programa."
Respuesta correcta:
{{"conceptos": []}}

══════════════════════════════════════════════════
FORMATO DE RESPUESTA (OBLIGATORIO)
══════════════════════════════════════════════════
Solo JSON válido. Sin markdown, sin texto adicional, sin explicaciones:
{{"conceptos": [{{"termino": "nombre_del_concepto", "definicion_video": "explicacion_exacta_extraida_del_video"}}]}}

Si no hay conceptos claros que extraer:
{{"conceptos": []}}

══════════════════════════════════════════════════
TRANSCRIPCIÓN A ANALIZAR:
══════════════════════════════════════════════════
{transcripcion}
"""

PROMPT_EVALUACION_INDIVIDUAL = """\
Eres un evaluador pedagógico ESTRICTO, DETERMINISTA y CON AMNESIA.
Tu ÚNICA fuente de verdad es el "Contexto oficial del grafo". 
PROHIBIDO usar tu conocimiento previo sobre programación o tecnología.

══════════════════════════════════════════════════
ESCALA DE EVALUACIÓN (5 niveles)
══════════════════════════════════════════════════
CORRECTO              (1.00) → Precisa, completa, sin errores factuales.
MAYORMENTE_CORRECTO   (0.75) → Idea principal correcta, omisiones menores.
PARCIAL               (0.50) → Idea principal correcta pero con imprecisiones no críticas.
MAYORMENTE_INCORRECTO (0.25) → Algo verdadero pero errores importantes dominan.
INCORRECTO            (0.00) → Errónea, contradice el oficial o invierte funcionalidades.

══════════════════════════════════════════════════
REGLAS ESTRICTAS
══════════════════════════════════════════════════
❌ PROHIBIDO usar PARCIAL o superior si:
   - La definición contradice directamente el conocimiento oficial.
   - Se atribuyen funcionalidades que pertenecen a otro concepto.
   - El tipo de dato u operación está intercambiado.
   - El núcleo del concepto es fundamentalmente erróneo.

✅ PARCIAL SOLO si:
   - La idea principal es correcta.
   - Solo faltan detalles secundarios o hay simplificación pedagógica razonable.

══════════════════════════════════════════════════
EJEMPLOS FEW-SHOT
══════════════════════════════════════════════════

Término: bool | Vídeo: "sirve para texto largo" | Oficial: "True/False, valores lógicos"
→ {{"etiqueta": "INCORRECTO", "score": 0.0, "justificacion": "Atribuye funcionalidad de str a bool.", "criterio_principal": "Error conceptual total: intercambio de tipos."}}

Término: int | Vídeo: "almacena decimales como 3.14" | Oficial: "números enteros sin decimal"
→ {{"etiqueta": "INCORRECTO", "score": 0.0, "justificacion": "Describe float, no int.", "criterio_principal": "Intercambio de funcionalidades entre tipos de dato."}}

Término: Variable | Vídeo: "guarda datos en el programa" | Oficial: "espacio en memoria con nombre que almacena un valor"
→ {{"etiqueta": "MAYORMENTE_CORRECTO", "score": 0.75, "justificacion": "Idea correcta, omite nombre y memoria.", "criterio_principal": "Omisiones menores sin error factual."}}

Término: float | Vídeo: "números con decimales" | Oficial: "tipo de punto flotante para números reales"
→ {{"etiqueta": "PARCIAL", "score": 0.5, "justificacion": "Correcto en lo esencial pero muy simplificado.", "criterio_principal": "Simplificación pedagógica sin error, omite punto flotante."}}

══════════════════════════════════════════════════
CONCEPTO A EVALUAR
══════════════════════════════════════════════════
Término              : {termino}
Definición del vídeo : {definicion_video}

Contexto oficial del grafo de conocimiento:
{contexto}

══════════════════════════════════════════════════
FORMATO DE RESPUESTA (OBLIGATORIO)
══════════════════════════════════════════════════
Solo JSON válido. Sin markdown, sin texto adicional:
{{"etiqueta": "CORRECTO|MAYORMENTE_CORRECTO|PARCIAL|MAYORMENTE_INCORRECTO|INCORRECTO", "score": 0.0, "justificacion": "...", "criterio_principal": "..."}}
"""

# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class ConceptoEvaluado(BaseModel):
    termino            : str
    definicion_video   : str
    etiqueta           : str   = Field(description="CORRECTO|MAYORMENTE_CORRECTO|PARCIAL|MAYORMENTE_INCORRECTO|INCORRECTO|SIN_EVIDENCIA")
    score              : float = Field(ge=0.0, le=1.0)
    justificacion      : str
    criterio_principal : str
    contexto_recuperado: str   = Field(description="Primeros 500 chars del contexto usado")

class ValidacionVideoResponse(BaseModel):
    video_id               : str
    score_global           : float = Field(ge=0.0, le=1.0)
    total_conceptos        : int
    correctos              : int
    mayormente_correctos   : int
    parciales              : int
    mayormente_incorrectos : int
    incorrectos            : int
    sin_evidencia          : int
    conceptos              : list[ConceptoEvaluado]
    tiempo_proceso_seg     : float

class ValidarVideoRequest(BaseModel):
    video_id     : str
    transcripcion: str

class ValidarVideosRequest(BaseModel):
    videos: list[ValidarVideoRequest]

class ConsultaRequest(BaseModel):
    pregunta: str

class ChatRequest(BaseModel):
    session_id: str
    mensaje   : str

# =============================================================================
# INFRAESTRUCTURA
# =============================================================================

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )

def init_graph_store() -> Neo4jPropertyGraphStore:
    return Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASS,
        url=NEO4J_URL,
    )

def _reset_chat_sessions():
    global chat_sessions
    chat_sessions = {}
    logger.info("🧹 Sesiones de chat reiniciadas.")

def _parse_json_seguro(texto: str, campo: Optional[str] = None) -> dict | list:
    """
    Parser robusto: limpia backticks, extrae bloque JSON balanceado,
    repara comillas duplicadas, loggea fragmento del texto corrupto.
    """
    if not texto:
        logger.warning("⚠️  Respuesta LLM vacía.")
        return [] if campo else {}

    # Limpiar markdown y backticks
    limpio = re.sub(r"```(?:json)?", "", texto, flags=re.IGNORECASE).strip()
    limpio = re.sub(r"```", "", limpio).strip()

    def _intentar(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            reparado = re.sub(r'""([^"]+)""', r'"\1"', s)
            try:
                return json.loads(reparado)
            except json.JSONDecodeError:
                return None

    # Intento directo
    resultado = _intentar(limpio)
    if resultado is not None:
        return resultado.get(campo, []) if campo else resultado

    # Extraer primer bloque JSON con llaves balanceadas
    inicio = limpio.find("{")
    if inicio != -1:
        profundidad = 0
        for i, ch in enumerate(limpio[inicio:], start=inicio):
            if ch == "{":
                profundidad += 1
            elif ch == "}":
                profundidad -= 1
                if profundidad == 0:
                    resultado = _intentar(limpio[inicio : i + 1])
                    if resultado is not None:
                        return resultado.get(campo, []) if campo else resultado
                    break

    logger.warning(f"⚠️  No se pudo parsear JSON. Primeros 300 chars: {texto[:300]!r}")
    return [] if campo else {}

def _es_contexto_util(contexto: str) -> bool:
    return bool(contexto and len(contexto.strip()) >= MIN_CONTEXT_LENGTH)

def _crear_concepto_sin_evidencia(termino: str, definicion: str) -> ConceptoEvaluado:
    return ConceptoEvaluado(
        termino=termino,
        definicion_video=definicion,
        etiqueta="SIN_EVIDENCIA",
        score=SCORE_MAP["SIN_EVIDENCIA"],
        justificacion=(
            "No se encontró información suficiente en el grafo de conocimiento "
            "para validar este concepto. No implica error en el vídeo."
        ),
        criterio_principal="Ausencia de contexto útil en el grafo de conocimiento.",
        contexto_recuperado="",
    )

# =============================================================================
# CARGA DEL ÍNDICE AL ARRANCAR
# =============================================================================
try:
    logger.info("🔍 Intentando cargar índice existente desde Neo4j...")
    _store = init_graph_store()
    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=_store,
        embed_model=Settings.embed_model,
        llm=query_llm,
    )
    logger.info("✅ Índice cargado desde Neo4j.")
except Exception as e:
    logger.warning(f"⚠️  No se pudo cargar índice: {e}")
    logger.warning("    Usa /cargar para construir el grafo desde MinIO.")

# =============================================================================
# PIPELINE: FUNCIONES ASÍNCRONAS
# =============================================================================

async def _recuperar_contexto(termino: str, definicion: str) -> str:
    """
    Retrieval (Neo4j, bloqueante → executor) + Rerank (CPU/GPU, bloqueante → executor).
    Devuelve "" si no hay contexto útil → activará SIN_EVIDENCIA sin llamar al LLM.
    """
    if graph_index is None:
        return ""

    try:
        query_texto = f"{termino}: {definicion}"
        retriever   = graph_index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
        loop        = asyncio.get_running_loop()

        nodos = await loop.run_in_executor(executor, retriever.retrieve, query_texto)
        if not nodos:
            logger.info(f"   ℹ️  Sin resultados en grafo para: '{termino}'")
            return ""

        try:
            query_bundle      = QueryBundle(query_str=query_texto)
            nodos_rerankeados = await loop.run_in_executor(
                executor,
                lambda: reranker.postprocess_nodes(nodos, query_bundle=query_bundle),
            )
            if not nodos_rerankeados:
                raise ValueError("Reranker vacío")
        except Exception as e_rerank:
            logger.warning(f"⚠️  Reranker falló para '{termino}': {e_rerank}. Usando top-1.")
            nodos_rerankeados = nodos[:1]

        contexto = "\n---\n".join(n.get_content() for n in nodos_rerankeados)
        logger.info(f"   ✅ '{termino}': {len(nodos_rerankeados)} fragmentos, {len(contexto)} chars")
        return contexto

    except Exception as e:
        logger.error(f"❌ Retrieval falló para '{termino}': {e}")
        return ""


async def _evaluar_concepto_individual(
    termino         : str,
    definicion_video: str,
    contexto        : str,   # [FIX 2] Ya viene truncado a MAX_CONTEXT_LENGTH
    video_id        : str,
) -> ConceptoEvaluado:
    """
    Evalúa UN concepto con acomplete nativo (no bloquea el event loop).
    La transcripción original NUNCA llega aquí.
    El contexto ya viene truncado a 4000 chars por el caller.
    """
    prompt = PROMPT_EVALUACION_INDIVIDUAL.format(
        termino=termino,
        definicion_video=definicion_video,
        contexto=contexto,
    )

    try:
        loop = asyncio.get_running_loop()
        respuesta = await loop.run_in_executor(executor, query_llm.complete, prompt)
        resultado = _parse_json_seguro(respuesta.text.strip())

        etiqueta = str(resultado.get("etiqueta", "INCORRECTO")).upper().strip()
        if etiqueta not in ETIQUETAS_VALIDAS - {"SIN_EVIDENCIA"}:
            logger.warning(
                f"[{video_id}] ⚠️  Etiqueta inválida '{etiqueta}' para '{termino}' → INCORRECTO"
            )
            etiqueta = "INCORRECTO"

        return ConceptoEvaluado(
            termino=termino,
            definicion_video=definicion_video,
            etiqueta=etiqueta,
            score=SCORE_MAP.get(etiqueta, 0.0),
            justificacion=resultado.get("justificacion", "Sin justificación."),
            criterio_principal=resultado.get("criterio_principal", "No especificado."),
            contexto_recuperado=contexto[:500],
        )

    except Exception as e:
        logger.error(f"[{video_id}] ❌ Error evaluando '{termino}': {e}")
        return ConceptoEvaluado(
            termino=termino,
            definicion_video=definicion_video,
            etiqueta="INCORRECTO",
            score=0.0,
            justificacion=f"Error interno en la evaluación: {e}",
            criterio_principal="Fallo del pipeline de evaluación.",
            contexto_recuperado=contexto[:500],
        )


async def _pipeline_validacion_async(video_id: str, transcripcion: str) -> ValidacionVideoResponse:
    """
    Pipeline de validación de un vídeo.

    Paso 1 — Extracción: acomplete con transcripción completa (UNA vez por vídeo).
    Paso 2 — Retrieval: asyncio.gather para todos los conceptos en paralelo.
    Paso 3 — Filtrado: SIN_EVIDENCIA automático si contexto insuficiente.
    Paso 4 — Evaluación: for-loop SECUENCIAL [FIX 1] para proteger la VRAM.
              Contexto truncado a MAX_CONTEXT_LENGTH [FIX 2] antes de cada llamada.
    Paso 5 — Métricas globales.
    """
    t_inicio = time.time()

    # ------------------------------------------------------------------
    # PASO 1: EXTRACCIÓN — acomplete, num_ctx=8192, transcripción completa
    # ------------------------------------------------------------------
    logger.info(f"[{video_id}] 📝 Paso 1/4: Extracción (acomplete)...")
    prompt_ext = PROMPT_EXTRACCION.format(transcripcion=transcripcion[:13_000])

    try:
        resp_ext      = await extractor_llm.acomplete(prompt_ext)
        conceptos_raw = _parse_json_seguro(resp_ext.text.strip(), "conceptos")
    except Exception as e:
        logger.error(f"[{video_id}] ❌ Error en extracción: {e}")
        conceptos_raw = []

    if not conceptos_raw:
        logger.warning(f"[{video_id}] ⚠️  Sin conceptos extraídos.")
        return ValidacionVideoResponse(
            video_id=video_id, score_global=0.0, total_conceptos=0,
            correctos=0, mayormente_correctos=0, parciales=0,
            mayormente_incorrectos=0, incorrectos=0, sin_evidencia=0,
            conceptos=[], tiempo_proceso_seg=round(time.time() - t_inicio, 2),
        )

    logger.info(f"[{video_id}] ✅ {len(conceptos_raw)} conceptos extraídos.")

    # ------------------------------------------------------------------
    # PASO 2: RETRIEVAL EN PARALELO (asyncio.gather — I/O bound, seguro)
    # Mientras Neo4j responde, el event loop atiende otros vídeos.
    # ------------------------------------------------------------------
    logger.info(f"[{video_id}] 🔍 Paso 2/4: Retrieval paralelo ({len(conceptos_raw)} conceptos)...")
    contextos: tuple[str, ...] = await asyncio.gather(*[
        _recuperar_contexto(c.get("termino", ""), c.get("definicion_video", ""))
        for c in conceptos_raw
    ])

    # ------------------------------------------------------------------
    # PASO 3: FILTRADO — SIN_EVIDENCIA sin gastar LLM
    # ------------------------------------------------------------------
    logger.info(f"[{video_id}] 🔎 Paso 3/4: Filtrando contextos...")

    con_contexto : list[tuple[dict, str]] = []
    sin_contexto : list[dict]             = []

    for c, ctx in zip(conceptos_raw, contextos):
        if _es_contexto_util(ctx):
            con_contexto.append((c, ctx))
        else:
            sin_contexto.append(c)

    logger.info(
        f"[{video_id}] → {len(con_contexto)} con contexto | "
        f"{len(sin_contexto)} SIN_EVIDENCIA (sin llamada LLM)"
    )

    # ------------------------------------------------------------------
    # PASO 4: EVALUACIÓN SECUENCIAL [FIX 1]
    # Un concepto a la vez → Ollama procesa una sola inferencia de evaluación
    # por vídeo en cada momento. Evita saturación de VRAM/KV Cache.
    # El paralelismo real viene del semáforo: 4 vídeos × 1 eval activa = 4 slots.
    # [FIX 2] ctx_seguro = ctx[:MAX_CONTEXT_LENGTH] protege num_ctx=4096.
    # ------------------------------------------------------------------
    conceptos_evaluados: list[ConceptoEvaluado] = []

    if con_contexto:
        logger.info(
            f"[{video_id}] 🎓 Paso 4/4: Evaluación secuencial "
            f"({len(con_contexto)} conceptos, ctx máx {MAX_CONTEXT_LENGTH} chars)..."
        )
        for idx, (c, ctx) in enumerate(con_contexto, start=1):
            termino  = c.get("termino", "")
            definicion = c.get("definicion_video", "")

            # [FIX 2] Truncar contexto antes de enviarlo al LLM
            ctx_seguro = ctx[:MAX_CONTEXT_LENGTH]

            logger.info(
                f"[{video_id}] → Evaluando {idx}/{len(con_contexto)}: "
                f"'{termino}' (ctx {len(ctx_seguro)} chars)"
            )

            resultado = await _evaluar_concepto_individual(
                termino=termino,
                definicion_video=definicion,
                contexto=ctx_seguro,
                video_id=video_id,
            )
            conceptos_evaluados.append(resultado)
    else:
        logger.info(f"[{video_id}] ⏭️  Paso 4/4: Saltado (ningún concepto con contexto útil)")

    # SIN_EVIDENCIA sin llamada al LLM
    for c in sin_contexto:
        conceptos_evaluados.append(
            _crear_concepto_sin_evidencia(
                c.get("termino", "desconocido"),
                c.get("definicion_video", ""),
            )
        )

    # ------------------------------------------------------------------
    # PASO 5: MÉTRICAS GLOBALES
    # ------------------------------------------------------------------
    total                  = len(conceptos_evaluados)
    correctos              = sum(1 for c in conceptos_evaluados if c.etiqueta == "CORRECTO")
    mayormente_correctos   = sum(1 for c in conceptos_evaluados if c.etiqueta == "MAYORMENTE_CORRECTO")
    parciales              = sum(1 for c in conceptos_evaluados if c.etiqueta == "PARCIAL")
    mayormente_incorrectos = sum(1 for c in conceptos_evaluados if c.etiqueta == "MAYORMENTE_INCORRECTO")
    incorrectos            = sum(1 for c in conceptos_evaluados if c.etiqueta == "INCORRECTO")
    sin_evidencia          = sum(1 for c in conceptos_evaluados if c.etiqueta == "SIN_EVIDENCIA")
    score_global           = round(sum(c.score for c in conceptos_evaluados) / total, 4) if total else 0.0
    elapsed                = round(time.time() - t_inicio, 2)

    logger.info(
        f"[{video_id}] ✅ Score: {score_global:.2f} | "
        f"✅{correctos} 🟩{mayormente_correctos} 🟡{parciales} "
        f"🟠{mayormente_incorrectos} ❌{incorrectos} ⬜{sin_evidencia} | {elapsed}s"
    )

    return ValidacionVideoResponse(
        video_id=video_id,
        score_global=score_global,
        total_conceptos=total,
        correctos=correctos,
        mayormente_correctos=mayormente_correctos,
        parciales=parciales,
        mayormente_incorrectos=mayormente_incorrectos,
        incorrectos=incorrectos,
        sin_evidencia=sin_evidencia,
        conceptos=conceptos_evaluados,
        tiempo_proceso_seg=elapsed,
    )


async def _pipeline_con_semaforo(video_id: str, transcripcion: str) -> ValidacionVideoResponse:
    """
    Wrapper de Nivel 1: controla cuántos vídeos acceden a la GPU simultáneamente.
    Los vídeos en espera ceden el event loop sin bloquear.
    """
    async with _video_semaphore:
        logger.info(f"[{video_id}] 🟢 Slot GPU adquirido ({VIDEO_CONCURRENCY} máx).")
        resultado = await _pipeline_validacion_async(video_id, transcripcion)
        logger.info(f"[{video_id}] 🔴 Slot GPU liberado.")
        return resultado

# =============================================================================
# CONSTRUCCIÓN DEL GRAFO (bloqueante → executor)
# =============================================================================

def _monitor_progreso(stop_event: threading.Event, mensaje: str, intervalo: int = 15):
    t0 = time.time()
    while not stop_event.is_set():
        time.sleep(intervalo)
        if not stop_event.is_set():
            logger.info(f"   ⏳ {mensaje} ... {time.time()-t0:.0f}s")

def _build_graph(documentos: list[Document]) -> PropertyGraphIndex:
    graph_store = init_graph_store()

    logger.info("📄 PASO 1/3: Chunking...")
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=64)
    nodos    = splitter.get_nodes_from_documents(documentos, show_progress=True)
    logger.info(f"✅ {len(nodos)} chunks.")

    logger.info(f"🧠 PASO 2/3: Extrayendo relaciones ({len(nodos)} chunks)...")
    extractor = SimpleLLMPathExtractor(
        llm=extractor_llm,
        max_paths_per_chunk=2,
        num_workers=MAX_WORKERS,
    )

    stop_event = threading.Event()
    monitor    = threading.Thread(
        target=_monitor_progreso,
        args=(stop_event, "Extrayendo relaciones", 15),
        daemon=True,
    )
    monitor.start()

    t1    = time.time()
    index = PropertyGraphIndex.from_documents(
        documentos,
        property_graph_store=graph_store,
        kg_extractors=[extractor],
        transformations=[splitter],
        embed_model=Settings.embed_model,
        llm=query_llm,
        show_progress=True,
    )
    stop_event.set()
    monitor.join()

    logger.info(f"✅ Grafo construido en {time.time()-t1:.1f}s")
    logger.info("💾 PASO 3/3: Guardado en Neo4j ✅")
    return index

def _procesar_un_documento(s3_client, bucket: str, key: str) -> str:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body     = response["Body"].read()
    if key.lower().endswith(".pdf"):
        doc   = fitz.open(stream=body, filetype="pdf")
        texto = chr(12).join([page.get_text() for page in doc])
        doc.close()
    else:
        texto = body.decode("utf-8", errors="ignore")
    return texto

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/cargar")
async def cargar_desde_minio():
    """Descarga documentos de MinIO, construye el grafo y lo persiste en Neo4j."""
    global graph_index
    try:
        s3      = get_s3_client()
        objetos = s3.list_objects_v2(Bucket=BUCKET_NAME).get("Contents", [])
        if not objetos:
            raise HTTPException(status_code=400, detail="Bucket vacío.")

        loop = asyncio.get_running_loop()
        logger.info(f"📥 Descargando {len(objetos)} archivos...")
        resultados = await asyncio.gather(*[
            loop.run_in_executor(executor, _procesar_un_documento, s3, BUCKET_NAME, obj["Key"])
            for obj in objetos
        ])

        documentos = [
            Document(text=texto, metadata={"fuente": obj["Key"]})
            for obj, texto in zip(objetos, resultados) if texto.strip()
        ]
        if not documentos:
            raise HTTPException(status_code=400, detail="No se encontró texto válido.")

        logger.info(f"🧠 Construyendo grafo con {len(documentos)} documentos...")
        graph_index = await loop.run_in_executor(executor, _build_graph, documentos)
        _reset_chat_sessions()

        return {"mensaje": "Grafo construido.", "documentos_procesados": len(documentos)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en carga: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consultar")
async def consultar(req: ConsultaRequest):
    """Consulta puntual al grafo sin memoria de conversación."""
    if graph_index is None:
        raise HTTPException(status_code=400, detail="Grafo no cargado.")
    try:
        query_engine = graph_index.as_query_engine(
            llm=query_llm,
            include_text=True,
            similarity_top_k=RETRIEVAL_TOP_K,
            node_postprocessors=[reranker],
        )
        loop      = asyncio.get_running_loop()
        respuesta = await loop.run_in_executor(executor, query_engine.query, req.pregunta)
        return {"pregunta": req.pregunta, "respuesta": str(respuesta)}
    except Exception as e:
        logger.error(f"❌ Error en consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat con memoria de conversación por sesión."""
    if graph_index is None:
        raise HTTPException(status_code=400, detail="Grafo no cargado.")
    try:
        if req.session_id not in chat_sessions:
            chat_sessions[req.session_id] = graph_index.as_chat_engine(
                llm=query_llm,
                chat_mode="condense_plus_context",
                similarity_top_k=RETRIEVAL_TOP_K,
                node_postprocessors=[reranker],
                verbose=False,
            )
            logger.info(f"💬 Nueva sesión: {req.session_id}")

        loop      = asyncio.get_running_loop()
        respuesta = await loop.run_in_executor(executor, chat_sessions[req.session_id].chat, req.mensaje)
        return {"session_id": req.session_id, "mensaje": req.mensaje, "respuesta": str(respuesta)}
    except Exception as e:
        logger.error(f"❌ Error en chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/{session_id}")
async def cerrar_sesion(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"mensaje": f"Sesión '{session_id}' cerrada."}
    raise HTTPException(status_code=404, detail="Sesión no encontrada.")


@app.post("/validar-video", response_model=ValidacionVideoResponse)
async def validar_video(req: ValidarVideoRequest):
    """Valida la precisión conceptual de UN vídeo. Usa el mismo semáforo que /validar-videos."""
    if graph_index is None:
        raise HTTPException(status_code=400, detail="Grafo no cargado.")
    if not req.transcripcion.strip():
        raise HTTPException(status_code=400, detail="Transcripción vacía.")
    try:
        return await _pipeline_con_semaforo(req.video_id, req.transcripcion)
    except Exception as e:
        logger.error(f"❌ Error validando '{req.video_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validar-videos", response_model=list[ValidacionVideoResponse])
async def validar_videos(req: ValidarVideosRequest):
    """
    Valida múltiples vídeos con paralelismo controlado:
      → gather lanza todos a la vez.
      → Semáforo(4) permite máximo 4 en GPU simultáneamente.
      → Dentro de cada vídeo: retrieval paralelo, evaluación secuencial.
      → Si un vídeo falla, devuelve score 0.0 y continúa con los demás.
    """
    if graph_index is None:
        raise HTTPException(status_code=400, detail="Grafo no cargado.")
    if not req.videos:
        raise HTTPException(status_code=400, detail="Lista de vídeos vacía.")

    logger.info(
        f"🚀 Lote de {len(req.videos)} vídeos | "
        f"Semáforo: {VIDEO_CONCURRENCY} slots GPU | "
        f"Evaluación: secuencial por vídeo | "
        f"ctx máx: {MAX_CONTEXT_LENGTH} chars"
    )

    try:
        resultados = await asyncio.gather(*[
            _pipeline_con_semaforo(v.video_id, v.transcripcion)
            for v in req.videos
        ], return_exceptions=True)

        respuestas: list[ValidacionVideoResponse] = []
        for v, resultado in zip(req.videos, resultados):
            if isinstance(resultado, Exception):
                logger.error(f"❌ Fallo total en '{v.video_id}': {resultado}")
                respuestas.append(ValidacionVideoResponse(
                    video_id=v.video_id, score_global=0.0, total_conceptos=0,
                    correctos=0, mayormente_correctos=0, parciales=0,
                    mayormente_incorrectos=0, incorrectos=0, sin_evidencia=0,
                    conceptos=[], tiempo_proceso_seg=0.0,
                ))
            else:
                respuestas.append(resultado)

        scores = [r.score_global for r in respuestas]
        logger.info(
            f"✅ Lote completo | {len(respuestas)} vídeos | "
            f"Score promedio: {sum(scores)/len(scores):.2f}"
        )
        return respuestas

    except Exception as e:
        logger.error(f"❌ Error en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/estado")
async def estado():
    return {
        "version"            : "5.1 — Semáforo + Eval Secuencial + ctx Safe",
        "grafo_cargado"      : graph_index is not None,
        "sesiones_activas"   : list(chat_sessions.keys()),
        "embed_device"       : _EMBED_DEVICE,
        "extractor_llm"      : EXTRACTOR_LLM_MODEL,
        "extractor_num_ctx"  : 8192,
        "query_llm"          : QUERY_LLM_MODEL,
        "query_num_ctx"      : 4096,
        "max_context_enviado": MAX_CONTEXT_LENGTH,
        "reranker"           : RERANKER_MODEL,
        "video_concurrency"  : VIDEO_CONCURRENCY,
        "retrieval_top_k"    : RETRIEVAL_TOP_K,
        "reranker_top_n"     : RERANKER_TOP_N,
        "min_context_length" : MIN_CONTEXT_LENGTH,
        "escala_scores"      : SCORE_MAP,
        "ollama_env"         : {
            "OLLAMA_NUM_PARALLEL"    : 8,
            "OLLAMA_MAX_LOADED_MODELS": 2,
        },
    }

# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)