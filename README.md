# MigraBot LLM
Asistente basado en Llama que ayuda a migrar bases de datos legacy (DBF) a PostgreSQL.

## Problema
Las empresas pierden semanas entendiendo tablas antiguas sin documentación antes de migrarlas.

## Solución y decisiones técnicas
- **Modelo:** Llama 3.2 3B ejecutado localmente con Ollama (sin API key).
- **RAG:** el diccionario de datos (`data/`) se divide en fragmentos, se convierte en embeddings multilingües (sentence-transformers) y se guarda en ChromaDB. En cada pregunta se recuperan los 3 fragmentos más relevantes.
- **Prompt engineering:** system prompt con rol, reglas para no inventar, formato fijo (Análisis / Script SQL / Validaciones) y un ejemplo few-shot.
- **Filtro anti-alucinación:** antes de llamar a Llama se valida que las tablas o campos mencionados existan en la documentación; si no existen, el asistente lo indica en lugar de inventar.
- **API:** FastAPI con endpoints `/consultar`, `/historial`, `/subir-archivo` y `/salud`, más una interfaz web de chat.

## Cómo ejecutarlo
1. Instalar Ollama y ejecutar `ollama pull llama3.2:3b`
2. `python -m venv venv` y activar el entorno
3. `pip install -r requirements.txt`
4. `uvicorn app:app` y abrir http://127.0.0.1:8000

## Ejemplos de preguntas
- ¿Cómo migro la tabla CLIENTES.DBF a PostgreSQL?
- ¿Qué hago con las fechas 00000000?
- ¿En qué orden cargo clientes y facturas?
- ¿Cómo migro la tabla PROVEEDORES? (responde que no hay documentación)