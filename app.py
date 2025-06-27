from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from src.extraction.unstructured_extraction import DocumentExtractor
from src.embedding.chroma_embedder import ChromaEmbedder
from src.rag_pipeline.retriever import Retriever
from src.rag_pipeline.context_builder import ContextBuilder
from src.rag_pipeline.llm_wrapper import LLMWrapper
from src.rag_pipeline.query_embedder import QueryEmbedder
from src.chunking.chunker import StructuredChunker

import uuid
import json
from pathlib import Path

app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Only allow your frontend
    allow_credentials=True,
    allow_methods=["POST"],  # or specify exact ones
    allow_headers=["Content-Type", "Authorization"],
)
DATA_DIR = Path("data")


@app.post("/new-session")
def new_session():
    session_id = str(uuid.uuid4())[:8]
    return {"session_id": session_id}


@app.post("/upload")
def upload(file: UploadFile = File(...), session_id: str = Form(...)):
    try:
        session_dir = DATA_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract
        extractor = DocumentExtractor(session_id=session_id, base_dir=DATA_DIR)
        extracted = extractor.process(file)

        # Step 2: Save raw extracted output to JSON
        json_path = session_dir / "extracted.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)

        # Step 3: Chunk using StructuredChunker
        chunker = StructuredChunker(input_path=json_path)
        chunks = chunker.chunk()
        chunked_json_path = session_dir/"chunked.json"
        with open(chunked_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        # Step 4: Embed chunks to Chroma
        embedder = ChromaEmbedder(chunk_json_path=chunked_json_path,collection_name=session_id)
        embedder.embed_and_store()

        return {
            "status": "success",
            "message": f"{len(chunks)} chunks embedded.",
            "chunks": len(chunks),
            "images": len(extracted["images"]),
            "tables": len(extracted["tables"]),
            "code_snippets": len(extracted["code_snippets"])
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/query")
def query(user_query: str = Form(...), session_id: str = Form(...)):
    try:
        # Step 1: Embed the query
        query_vector = QueryEmbedder().embed(user_query)

        # Step 2: Retrieve top chunks
        retriever = Retriever(collection_name=session_id)
        top_chunks = retriever.retrieve(query_embedding=query_vector, top_k=5)

        # Step 3: Build context
        builder = ContextBuilder()
        context_data = builder.build(top_chunks)

        # Step 4: LLM
        system_prompt = """
You are a helpful assistant that provides clear, well-structured, and well-formatted answers based on the given context.

Always follow these rules:
- Use Markdown formatting throughout.
- Structure the response into numbered major topics and subtopics based on core concepts, not just listing content types.
- Integrate **all image references** directly in the explanation using `![Alt Text](path)` and provide descriptive captions and contextual relevance.
- Include all **tables** and **code snippets** exactly as provided, embedding them naturally where they support the explanation.
- Use bullet points or numbered lists to organize details within subtopics.
- Use math notation (LaTeX style) where appropriate for equations.
- Be concise yet thorough, ensuring no referenced visual or code element is omitted.
- Write in a conversational, academic style suitable for graduate-level readers.

Never treat images, tables, or code snippets as separate sections; always embed them smoothly into the flow of your explanation.
"""

        llm = LLMWrapper()
        result = llm.query(
            user_query=user_query,
            context=context_data["context"],
            image_refs=context_data["images"],
            table_refs=context_data["tables"],
            code_snippets=context_data["code"],
            system_prompt=system_prompt
        )

        return {
            "status": "success",
            "response": result
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}
