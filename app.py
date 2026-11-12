"""MigraBot LLM: asistente de migración de bases de datos legacy con Llama (Ollama) + RAG."""
import os
import re

import chromadb
import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODELO_LLAMA = "llama3.2:3b"          # si tu PC es lenta, usa "llama3.2:1b"
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"  # entiende español
CARPETA_DATOS = "data"

app = FastAPI(title="MigraBot LLM API",
              description="Asistente con Llama + RAG para migrar bases de datos legacy a PostgreSQL.")

# ---------- RAG: embeddings + base vectorial ----------
print("Cargando modelo de embeddings (la primera vez se descarga, espera 1-2 min)...")
embedder = SentenceTransformer(MODELO_EMBEDDINGS)
cliente_chroma = chromadb.PersistentClient(path="chroma_db")
coleccion = cliente_chroma.get_or_create_collection(name="migracion", embedding_function=None)


def fragmentar(texto):
    """Divide el documento en bloques separados por líneas en blanco."""
    return [b.strip() for b in texto.split("\n\n") if b.strip()]


def indexar(texto, fuente):
    fragmentos = fragmentar(texto)
    if not fragmentos:
        return 0
    coleccion.upsert(
        ids=[f"{fuente}-{i}" for i in range(len(fragmentos))],
        documents=fragmentos,
        embeddings=embedder.encode(fragmentos).tolist(),
        metadatas=[{"fuente": fuente} for _ in fragmentos],
    )
    return len(fragmentos)


def buscar_contexto(pregunta, k=3):
    total = coleccion.count()
    if total == 0:
        return []
    r = coleccion.query(query_embeddings=embedder.encode([pregunta]).tolist(),
                        n_results=min(k, total))
    return [{"texto": d, "fuente": m["fuente"]}
            for d, m in zip(r["documents"][0], r["metadatas"][0])]

def terminos_desconocidos(pregunta):
    """Detecta tablas o campos mencionados en la pregunta que no existen en la documentación."""
    candidatos = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", pregunta))
    candidatos |= {t.upper() for t in re.findall(r"(?:tabla|campo)\s+(\w+)", pregunta, flags=re.IGNORECASE)}
    documentacion = " ".join(coleccion.get()["documents"]).upper()
    return sorted(c for c in candidatos if c not in documentacion)

os.makedirs(CARPETA_DATOS, exist_ok=True)
for nombre_archivo in os.listdir(CARPETA_DATOS):
    if nombre_archivo.endswith(".txt"):
        with open(os.path.join(CARPETA_DATOS, nombre_archivo), encoding="utf-8") as f:
            n = indexar(f.read(), nombre_archivo)
        print(f"Indexado {nombre_archivo}: {n} fragmentos")

# ---------- Prompt engineering ----------
SYSTEM_PROMPT = """Eres MigraBot, ingeniero de datos experto en migrar bases de datos legacy (DBF) a PostgreSQL.

REGLAS:
1. Usa ÚNICAMENTE la información del CONTEXTO para nombres de tablas, campos y reglas de negocio.
2. Si el contexto no contiene la respuesta, di: "No encuentro esa información en la documentación cargada." No inventes campos.
3. Responde en español con este formato:
**Análisis:** 2 a 3 líneas.
**Script SQL (PostgreSQL):** un bloque de código sql.
**Validaciones:** lista corta de consultas para verificar la migración.

EJEMPLO:
Pregunta: ¿Cómo convierto el campo FEC_FAC?
Respuesta:
**Análisis:** FEC_FAC viene como texto DDMMAAAA y en PostgreSQL debe ser DATE. Las fechas '00000000' se cargan como NULL.
**Script SQL (PostgreSQL):**
```sql
SELECT CASE WHEN fec_fac IN ('00000000', '') THEN NULL
            ELSE TO_DATE(fec_fac, 'DDMMYYYY') END AS fecha
FROM staging_facturas;
```
**Validaciones:**
- SELECT COUNT(*) FROM facturas WHERE fecha IS NULL;
"""


class Consulta(BaseModel):
    pregunta: str


historial = []


@app.get("/", include_in_schema=False)
def inicio():
    return RedirectResponse("/static/index.html")


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo": MODELO_LLAMA, "fragmentos_indexados": coleccion.count()}


@app.post("/consultar")
def consultar(consulta: Consulta):
    if not consulta.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta está vacía.")

    # Filtro anti-alucinación: si pregunta por tablas o campos que no están documentados, no se llama a Llama
    desconocidos = terminos_desconocidos(consulta.pregunta)
    if desconocidos:
        registro = {
            "pregunta": consulta.pregunta,
            "respuesta": (f"No encuentro información sobre {', '.join(desconocidos)} en la documentación cargada. "
                          "Sube su diccionario de datos con /subir-archivo para poder ayudarte."),
            "fuentes": [],
            "contexto": [],
        }
        historial.append(registro)
        return registro

    contexto = buscar_contexto(consulta.pregunta)
    texto_contexto = "\n\n".join(c["texto"] for c in contexto) or "Sin documentos cargados."

    try:
        r = ollama.chat(
            model=MODELO_LLAMA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXTO:\n{texto_contexto}\n\nPREGUNTA: {consulta.pregunta}"},
            ],
            options={"temperature": 0.2, "num_predict": 350},
        )
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"No se pudo usar Ollama. Revisa que esté abierto y que descargaste {MODELO_LLAMA}. Detalle: {e}")

    registro = {
        "pregunta": consulta.pregunta,
        "respuesta": r["message"]["content"],
        "fuentes": sorted({c["fuente"] for c in contexto}),
        "contexto": [c["texto"] for c in contexto],
    }
    historial.append(registro)
    return registro


@app.get("/historial")
def ver_historial():
    return {"total_consultas": len(historial), "historial": historial}


@app.post("/subir-archivo")
async def subir_archivo(archivo: UploadFile = File(...)):
    if not archivo.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .txt")
    contenido = (await archivo.read()).decode("utf-8", errors="ignore")
    nombre = os.path.basename(archivo.filename)
    with open(os.path.join(CARPETA_DATOS, nombre), "w", encoding="utf-8") as f:
        f.write(contenido)
    n = indexar(contenido, nombre)
    return {"mensaje": f"'{nombre}' indexado con {n} fragmentos."}


app.mount("/static", StaticFiles(directory="static"), name="static")