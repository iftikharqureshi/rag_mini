# RAG Mini Demo

A short notebook that builds a minimal retrieval-augmented QA flow over a local text file using OpenAI embeddings and chat completions. The goal is to let a reviewer see the full pipeline—load, chunk, embed, retrieve, and answer—inside a single, readable notebook.

## What’s here
- `rag.ipynb`: end-to-end notebook defining the `DocumentSearch` class, building embeddings, and answering two sample questions.
- `my_documents/ik.txt`: source text (Imran Khan Wikipedia excerpt) used to generate chunks and embeddings.

## How it works
- Documents are loaded from `my_documents`, split into ~1,000-character chunks, embedded with `text-embedding-ada-002`, and cached in memory.
- Queries are embedded the same way; cosine similarity ranks the chunks.
- Top chunks are passed to a `gpt-5-mini` chat completion to generate the final answer with the retrieved context inline.

## Running the notebook
1) Create a virtual environment (optional): `python -m venv .venv && source .venv/bin/activate`
2) Install deps: `pip install openai numpy jupyter`
3) Set your API key: `export OPENAI_API_KEY=your_key`
4) Launch: `jupyter notebook rag.ipynb`
5) Run cells top-to-bottom. The sample questions (“When was Imran Khan born?” and “When did he win the world cup?”) demonstrate retrieval + answer generation.

## Reviewing
- Focus on the `DocumentSearch` class in `rag.ipynb`: `_load_and_embed_docs`, `ask`, and `answer_question` are the core flow.
- Data is local only; swapping in other `.txt` files in `my_documents/` will reuse the same pipeline.
- Error handling is minimal; API failures surface as returned strings from `answer_question`.

## Notes and limitations
- Embeddings are recomputed on each run (no persistence layer).
- Chunking is naive word-based segmentation; no overlap or tokenizer-aware sizing.
- Costs and latency depend on OpenAI API usage; the notebook makes live calls.
