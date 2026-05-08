import nest_asyncio
nest_asyncio.apply()

import asyncio
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import boto3
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL  = "http://localhost:11434"
MINIO_ENDPOINT   = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME      = "rag-graph-aprendia"
NEO4J_URL        = "bolt://localhost:7687"
NEO4J_USER       = "neo4j"
NEO4J_PASS       = "password123"

DEFAULT_LLM_MODEL   = "llama3.1"
DEFAULT_EMBED_MODEL = "bge-m3"

RETRIEVAL_TOP_K = 3
RERANKER_TOP_N  = 1
CHUNK_MAX_CHARS = 400
MAX_WORKERS     = 8

SCORE_MAPEO = {
    "incorrecto": 0.2,
    "parcial":    0.5,
    "correcto":   0.8,
    "perfecto":   1.0,
}

EVAL_DATASET = [
    {"question": "¿Qué es Python?",
     "ground_truth": "Es un lenguaje de programación interpretado, de alto nivel y de propósito general, diseñado para ser fácil de leer y escribir."},
    {"question": "¿Qué es una variable en Python?",
     "ground_truth": "Es un nombre que se usa para almacenar un valor en memoria y poder reutilizarlo en el programa."},
    {"question": "¿Qué es un tipo de dato en Python?",
     "ground_truth": "Es la clasificación del valor que puede almacenar una variable, como enteros, cadenas o booleanos."},
    {"question": "¿Qué es una función en Python?",
     "ground_truth": "Es un bloque de código reutilizable que realiza una tarea específica y puede recibir parámetros y devolver resultados."},
    {"question": "¿Qué es una lista en Python?",
     "ground_truth": "Es una estructura de datos que permite almacenar múltiples elementos en una colección ordenada y mutable."},
]

# ---------------------------------------------------------------------------
# APP + GLOBAL STATE
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Grafo", version="5.0")

graph_index   : Optional[PropertyGraphIndex] = None
_query_engine : Optional[BaseQueryEngine]    = None   # cached, built once per index
_retriever = None


executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Retrieval cache: (concepto_key, definicion_key) → contexto str | None
_retrieval_cache : dict[tuple[str, str], Optional[str]] = {}
# Concept extraction cache: md5(transcripcion) → list[dict]
_concept_cache   : dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# LLM  — fully deterministic
# ---------------------------------------------------------------------------
Settings.llm = Ollama(
    model=DEFAULT_LLM_MODEL,
    temperature=0.0,
    additional_kwargs={"top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0},
    request_timeout=300.0,
    base_url=OLLAMA_BASE_URL,
)
Settings.embed_model = OllamaEmbedding(
    model_name=DEFAULT_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# ---------------------------------------------------------------------------
# RERANKER  — loaded ONCE globally, GPU-aware
# ---------------------------------------------------------------------------
try:
    import torch
    _RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _RERANKER_DEVICE = "cpu"

_RERANKER = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=RERANKER_TOP_N,
    device=_RERANKER_DEVICE,
)
print(f"✅ Reranker cargado en: {_RERANKER_DEVICE}")

# ---------------------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------------------
class PreguntaDTO(BaseModel):
    pregunta: str

class TextoDTO(BaseModel):
    texto: str

class VideoInput(BaseModel):
    id: str
    transcripcion: str

class MultiVideoDTO(BaseModel):
    videos: list[VideoInput]

class VideoTranscripcionDTO(BaseModel):
    transcripcion: str

class ConceptoEvaluado(BaseModel):
    concepto: str
    definicion_video: str
    contexto_rag: str
    score: float
    justificacion: str

class ResumenEvaluacion(BaseModel):
    conceptos_correctos: int
    conceptos_incorrectos: int
    nivel_general: str

class ValidacionVideoResponse(BaseModel):
    total_conceptos: int
    score_final: float
    evaluaciones: list[ConceptoEvaluado]
    resumen: ResumenEvaluacion

class MultiVideoResponse(BaseModel):
    resultados: dict[str, ValidacionVideoResponse]

# ---------------------------------------------------------------------------
# INFRASTRUCTURE HELPERS
# ---------------------------------------------------------------------------
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

def _require_index() -> None:
    if graph_index is None or _query_engine is None:
        raise HTTPException(
            status_code=400,
            detail="El índice no está listo. Ejecuta /cargar o /cargartexto primero.",
        )

# ---------------------------------------------------------------------------
# QUERY ENGINE  — built once per index load, reused globally
# ---------------------------------------------------------------------------
def _reset_global_state() -> None:
    """Rebuilds cached query engine and clears all caches after a new index is loaded."""
    global _query_engine, _retriever

    assert graph_index is not None
    _query_engine = graph_index.as_query_engine(
        include_text=True,
        similarity_top_k=RETRIEVAL_TOP_K,
        node_postprocessors=[_RERANKER],
        response_mode="no_text",
    )
    _retrieval_cache.clear()
    _concept_cache.clear()
     # 🔥 NUEVO: retriever global
    _retriever = graph_index.as_retriever(similarity_top_k=2)

    print("✅ Query engine (re)inicializado y caches limpiados.")

# ---------------------------------------------------------------------------
# INDEX BOOTSTRAP
# ---------------------------------------------------------------------------
try:
    _store = init_graph_store()
    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=_store,
        embed_model=Settings.embed_model,
    )
    _reset_global_state()
    print("✅ Índice cargado desde Neo4j.")
except Exception as e:
    print(f"⚠️  En espera de /cargar o /cargartexto: {e}")

# ---------------------------------------------------------------------------
# GRAPH BUILDER
# ---------------------------------------------------------------------------
def _build_graph(documentos: list[Document]) -> PropertyGraphIndex:
    graph_store = init_graph_store()
    splitter    = SentenceSplitter(chunk_size=1024, chunk_overlap=64)
    extractor   = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=5,
        num_workers=1,
    )
    return PropertyGraphIndex.from_documents(
        documentos,
        property_graph_store=graph_store,
        kg_extractors=[extractor],
        transformations=[splitter],
        embed_model=Settings.embed_model,
        show_progress=True,
    )

# ---------------------------------------------------------------------------
# ENDPOINTS: /cargartexto  /cargar  /preguntar  /estado  /evaluar
# ---------------------------------------------------------------------------
@app.post("/cargartexto")
async def cargar_texto(data: TextoDTO):
    global graph_index
    if not data.texto.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")
    try:
        print(f"Texto recibido ({len(data.texto)} chars).")
        loop        = asyncio.get_running_loop()
        graph_index = await loop.run_in_executor(
            executor, _build_graph,
            [Document(text=data.texto, metadata={"fuente": "input_manual"})]
        )
        _reset_global_state()
        return {"mensaje": "Grafo construido exitosamente.", "caracteres": len(data.texto)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cargar")
async def cargar_desde_minio():
    global graph_index
    try:
        s3      = get_s3_client()
        objetos = s3.list_objects_v2(Bucket=BUCKET_NAME).get("Contents", [])
        if not objetos:
            raise HTTPException(status_code=400, detail="Bucket vacío.")

        documentos: list[Document] = []
        for obj in objetos:
            key  = obj["Key"]
            body = s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()
            texto = (
                "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(body)).pages)
                if key.lower().endswith(".pdf")
                else body.decode("utf-8", errors="ignore")
            )
            if texto.strip():
                documentos.append(Document(text=texto, metadata={"fuente": key}))

        if not documentos:
            raise HTTPException(status_code=400, detail="No se encontró texto válido en el bucket.")

        loop        = asyncio.get_running_loop()
        graph_index = await loop.run_in_executor(executor, _build_graph, documentos)
        _reset_global_state()
        return {"mensaje": "Grafo construido exitosamente.", "documentos_procesados": len(documentos)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preguntar")
async def preguntar(data: PreguntaDTO):
    _require_index()
    if not data.pregunta.strip():
        raise HTTPException(status_code=400, detail="Pregunta vacía.")
    try:
        # Synthesis mode (different from validation engine)
        qe        = graph_index.as_query_engine(include_text=True, similarity_top_k=RETRIEVAL_TOP_K)
        loop      = asyncio.get_running_loop()
        respuesta = await loop.run_in_executor(executor, qe.query, data.pregunta)
        return {"pregunta": data.pregunta, "respuesta": str(respuesta)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/estado")
def estado():
    return {
        "indice_cargado":       graph_index is not None,
        "listo_para_consultas": graph_index is not None and _query_engine is not None,
    }


@app.post("/evaluar")
async def evaluar():
    _require_index()
    qe   = graph_index.as_query_engine(include_text=True, similarity_top_k=RETRIEVAL_TOP_K)
    loop = asyncio.get_running_loop()
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in EVAL_DATASET:
        try:
            result       = await loop.run_in_executor(executor, qe.query, item["question"])
            source_nodes = getattr(result, "source_nodes", [])
            ctx          = [n.get_content() for n in source_nodes] if source_nodes else [str(result)]
            questions.append(item["question"]); answers.append(str(result))
            contexts.append(ctx);              ground_truths.append(item["ground_truth"])
        except Exception as e:
            print(f"⚠️ RAGAS error en '{item['question']}': {e}")
            questions.append(item["question"]); answers.append("")
            contexts.append([""]);             ground_truths.append(item["ground_truth"])

    hf_dataset = Dataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts,  "ground_truth": ground_truths,
    })
    ragas_llm = LangchainLLMWrapper(
        ChatOllama(model=DEFAULT_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    )
    ragas_emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb

    result_ragas = await loop.run_in_executor(
        executor, lambda: evaluate(hf_dataset, metrics=metrics)
    )
    scores_df   = result_ragas.to_pandas()
    col_q       = "user_input" if "user_input" in scores_df.columns else "question"
    col_a       = "response"   if "response"   in scores_df.columns else "answer"
    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                   if c in scores_df.columns]

    return {
        "resumen":              {m: round(scores_df[m].mean(), 4) for m in metric_cols},
        "detalle_por_pregunta": scores_df[[col_q, col_a] + metric_cols].to_dict(orient="records"),
    }

# ---------------------------------------------------------------------------
# STEP 1 — CONCEPT EXTRACTION  (1 LLM call, cached by content hash)
# ---------------------------------------------------------------------------
def _hash_text(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

def _extraer_conceptos(transcripcion: str) -> list[dict]:
    key = _hash_text(transcripcion)
    if key in _concept_cache:
        return _concept_cache[key]

    prompt = (
        "Eres un experto en análisis de contenido educativo.\n"
        "Analiza la transcripción y extrae TODOS los conceptos teóricos o técnicos relevantes.\n\n"
        "Para cada concepto proporciona:\n"
        '- "concepto": nombre exacto del concepto\n'
        '- "definicion_video": definición o explicación dada en el video\n\n'
        "Responde ÚNICAMENTE con un array JSON válido. Sin texto adicional, sin markdown.\n"
        'Ejemplo: [{"concepto": "variable", "definicion_video": "contenedor que almacena datos"}]\n'
        "Si no hay conceptos relevantes, responde: []\n\n"
        f"TRANSCRIPCIÓN:\n{transcripcion}\n\nJSON:"
    )

    raw    = str(Settings.llm.complete(prompt)).strip()
    raw    = re.sub(r"```(?:json)?|```", "", raw).strip()
    result : list[dict] = []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = parsed
    except Exception:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except Exception:
                pass

    result = [
        c for c in result
        if isinstance(c, dict)
        and str(c.get("concepto", "")).strip()
        and str(c.get("definicion_video", "")).strip()
    ]
    _concept_cache[key] = result
    return result

# ---------------------------------------------------------------------------
# STEP 2 — RETRIEVAL  (cached, uses global query engine + global reranker)
# ---------------------------------------------------------------------------
def _recuperar_contexto_ant(concepto: str, definicion: str) -> Optional[str]:
    """
    Returns context string from RAG graph, or None if nothing relevant found.
    Concept + definition used together as query for better semantic matching.
    """
    cache_key = (concepto.lower().strip(), definicion.lower().strip()[:120])
    if cache_key in _retrieval_cache:
        return _retrieval_cache[cache_key]

    assert _query_engine is not None
    result       = _query_engine.query(f"{concepto}: {definicion}")
    source_nodes = getattr(result, "source_nodes", [])

    if source_nodes:
        chunks   = [n.get_content()[:CHUNK_MAX_CHARS].strip() for n in source_nodes]
        contexto : Optional[str] = "\n\n".join(chunks)
    else:
        fallback = str(result).strip()[:CHUNK_MAX_CHARS]
        contexto  = fallback if fallback else None

    _retrieval_cache[cache_key] = contexto
    return contexto

def _recuperar_contexto(concepto: str) -> Optional[str]:
    key = concepto.lower().strip()

    if key in _retrieval_cache:
        return _retrieval_cache[key]

    assert _retriever is not None

    nodes = _retriever.retrieve(key)

    if not nodes:
        _retrieval_cache[key] = None
        return None

    contexto = nodes[0].get_content()[:CHUNK_MAX_CHARS].strip()

    _retrieval_cache[key] = contexto
    return contexto

# ---------------------------------------------------------------------------
# STEP 3 — BATCH LLM EVALUATION  (1 call for all valid-context concepts)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
Eres un evaluador experto en contenido educativo.

Tu tarea es evaluar definiciones dadas por un autor en un video.

IMPORTANTE:
- SOLO evalúas conceptos que tienen contexto en el grafo de conocimiento
- Si no hay contexto, ese concepto NO se evalúa
- NO inventes información

REGLAS ESTRICTAS:
1. Si la definición contradice el contexto → "incorrecto"
2. Si falta información clave importante → "parcial"
3. Si es correcta aunque simplificada → "correcto"
4. Si es completa y precisa → "perfecto"

REGLAS ADICIONALES:
- Prefiere "correcto" sobre "parcial" si la idea principal está bien

Responde SOLO JSON válido.\
"""

# - Si no hay contradicción, NO uses "incorrecto"
# - Mantén el mismo criterio para todos los conceptos

def _evaluar_conceptos_batch(items: list[dict]) -> list[dict]:
    """
    items: [{"id": int, "concepto": str, "definicion_video": str, "contexto_rag": str}]
    Returns: [{"id": int, "label": str, "justificacion": str}]
    One single LLM call for all concepts.
    """
    if not items:
        return []

    payload = json.dumps(items, ensure_ascii=False, indent=2)
    prompt  = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Evalúa los siguientes {len(items)} conceptos:\n\n"
        f"{payload}\n\n"
        f"Responde con un array JSON de exactamente {len(items)} objetos en el mismo orden.\n"
        'Formato: [{"id": 0, "label": "correcto", "justificacion": "explicación breve en español"}]\n\n'
        "JSON:"
    )

    raw = str(Settings.llm.complete(prompt)).strip()
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    def _parse(text: str) -> list[dict]:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected list")
        out = []
        for entry in data:
            label = str(entry.get("label", "")).strip().lower()
            if label not in SCORE_MAPEO:
                label = "incorrecto"
            out.append({
                "id":            int(entry.get("id", -1)),
                "label":         label,
                "justificacion": str(entry.get("justificacion", "")),
            })
        return out

    # Attempt 1: direct parse
    try:
        return _parse(raw)
    except Exception:
        pass

    # Attempt 2: extract first JSON array
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return _parse(match.group())
        except Exception:
            pass

    # Fallback
    print(f"⚠️ Batch eval parse failed. Raw[:400]: {raw[:400]}")
    return [
        {"id": it["id"], "label": "incorrecto", "justificacion": "Error al parsear respuesta del LLM."}
        for it in items
    ]

# ---------------------------------------------------------------------------
# STEP 4 — AGGREGATION
# ---------------------------------------------------------------------------
def _agregar_resultados(evaluaciones: list[ConceptoEvaluado]) -> tuple[float, ResumenEvaluacion]:
    if not evaluaciones:
        return 0.0, ResumenEvaluacion(
            conceptos_correctos=0, conceptos_incorrectos=0, nivel_general="bajo"
        )
    scores      = [e.score for e in evaluaciones]
    score_final = round(sum(scores) / len(scores), 4)
    correctos   = sum(1 for s in scores if s >= 0.7)
    nivel       = "alto" if score_final >= 0.75 else ("medio" if score_final >= 0.45 else "bajo")
    return score_final, ResumenEvaluacion(
        conceptos_correctos=correctos,
        conceptos_incorrectos=len(scores) - correctos,
        nivel_general=nivel,
    )

# ---------------------------------------------------------------------------
# FULL PIPELINE  (single video)
# ---------------------------------------------------------------------------
async def _pipeline_validacion_async(transcripcion: str) -> ValidacionVideoResponse:
    loop = asyncio.get_running_loop()

    # ── STEP 1: Extract concepts (1 LLM call) ───────────────────────────────
    print("[pipeline] Extrayendo conceptos...")
    conceptos_raw: list[dict] = await loop.run_in_executor(
        executor, _extraer_conceptos, transcripcion
    )
    if not conceptos_raw:
        print("[pipeline] Sin conceptos extraídos.")
        score_final, resumen = _agregar_resultados([])
        return ValidacionVideoResponse(
            total_conceptos=0, score_final=score_final,
            evaluaciones=[], resumen=resumen,
        )

    print(f"[pipeline] {len(conceptos_raw)} conceptos. Retrieval en paralelo...")

    # ── STEP 2: Parallel retrieval ───────────────────────────────────────────
    retrieval_tasks = [
        loop.run_in_executor(
            executor,
            _recuperar_contexto,
            c["concepto"],
        )
        for c in conceptos_raw
    ]
    retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

    evaluaciones : list[ConceptoEvaluado] = []
    to_evaluate  : list[dict]             = []   # only concepts with valid context

    for idx, (concepto_dict, ctx_result) in enumerate(zip(conceptos_raw, retrieval_results)):
        concepto       = concepto_dict["concepto"]
        definicion_vid = concepto_dict["definicion_video"]

        # No context or retrieval error → score 0.0, skip LLM
        if isinstance(ctx_result, Exception) or not ctx_result:
            print(f"[pipeline] '{concepto}': sin contexto → score=0.0")
            evaluaciones.append(ConceptoEvaluado(
                concepto=concepto,
                definicion_video=definicion_vid,
                contexto_rag="Concepto no encontrado en el grafo de conocimiento.",
                score=0.0,
                justificacion="Concepto no encontrado en el grafo de conocimiento.",
            ))
        else:
            to_evaluate.append({
                "id":              idx,
                "concepto":        concepto,
                "definicion_video": definicion_vid,
                "contexto_rag":    ctx_result,
            })

    # ── STEP 3: Single batch LLM evaluation (concepts with context only) ────
    if to_evaluate:
        print(f"[pipeline] Evaluando {len(to_evaluate)} conceptos en 1 llamada LLM...")
        eval_results: list[dict] = await loop.run_in_executor(
            executor, _evaluar_conceptos_batch, to_evaluate
        )
        eval_by_id = {r["id"]: r for r in eval_results}

        for item in to_evaluate:
            idx           = item["id"]
            eval_res      = eval_by_id.get(idx, {"label": "incorrecto", "justificacion": "Sin resultado."})
            label         = eval_res["label"]
            score         = SCORE_MAPEO.get(label, 0.2)   # pure LLM score — no similarity
            justificacion = eval_res["justificacion"]
            print(f"[pipeline]   '{item['concepto']}': {label} → {score}")

            evaluaciones.append(ConceptoEvaluado(
                concepto=item["concepto"],
                definicion_video=item["definicion_video"],
                contexto_rag=item["contexto_rag"],
                score=score,
                justificacion=f"[{label}] {justificacion}",
            ))

    # ── STEP 4: Aggregate ────────────────────────────────────────────────────
    score_final, resumen = _agregar_resultados(evaluaciones)
    print(f"[pipeline] Score final: {score_final} | Nivel: {resumen.nivel_general}")

    return ValidacionVideoResponse(
        total_conceptos=len(evaluaciones),
        score_final=score_final,
        evaluaciones=evaluaciones,
        resumen=resumen,
    )

# ---------------------------------------------------------------------------
# ENDPOINTS: /validar-video  /validar-videos
# ---------------------------------------------------------------------------
@app.post("/validar-video", response_model=ValidacionVideoResponse)
async def validar_video(data: VideoTranscripcionDTO):
    """Validates a single video transcription against the knowledge graph."""
    _require_index()
    if not data.transcripcion.strip():
        raise HTTPException(status_code=400, detail="La transcripción está vacía.")
    print(f"[/validar-video] {len(data.transcripcion)} chars.")
    try:
        return await _pipeline_validacion_async(data.transcripcion)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[/validar-video] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validar-videos", response_model=MultiVideoResponse)
async def validar_videos(data: MultiVideoDTO):
    """Validates multiple video transcriptions in parallel."""
    _require_index()
    if not data.videos:
        raise HTTPException(status_code=400, detail="Lista de videos vacía.")

    empty = [v.id for v in data.videos if not v.transcripcion.strip()]
    if empty:
        raise HTTPException(status_code=400, detail=f"Transcripción vacía en: {empty}")

    print(f"[/validar-videos] {len(data.videos)} videos en paralelo...")
    tasks   = [_pipeline_validacion_async(v.transcripcion) for v in data.videos]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output: dict[str, ValidacionVideoResponse] = {}
    for video, result in zip(data.videos, results):
        if isinstance(result, Exception):
            print(f"[/validar-videos] ERROR en '{video.id}': {result}")
            output[video.id] = ValidacionVideoResponse(
                total_conceptos=0, score_final=0.0, evaluaciones=[],
                resumen=ResumenEvaluacion(
                    conceptos_correctos=0, conceptos_incorrectos=0, nivel_general="bajo"
                ),
            )
        else:
            output[video.id] = result

    return MultiVideoResponse(resultados=output)


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)