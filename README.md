
# FlashFind — Local AI Semantic Search for Your Files

A tiny **RAG‑lite** app that lets you drop PDFs and text files into a folder, ask questions in plain English, and get the most relevant passages back with a one‑paragraph summary. Runs fully on your CPU. No API keys, no billing.

## ✨ Features
- **Semantic search** using `all-MiniLM-L6-v2` sentence embeddings
- **Sources with exact passages** so you can verify results
- **Auto‑summary** (lightweight TextRank) for quick takeaways
- **Works offline** after the first model download
- **Zero cost**: Streamlit + sentence-transformers + pypdf

## 🧱 Project Structure
```
flashfind/
  app.py
  ingest.py
  utils.py
  requirements.txt
  data/          # put PDFs, .txt, .md here (not committed)
  index/         # auto-created; stores embeddings
  README.md
```

## 🚀 Quickstart
1. **Clone & setup**
   ```bash
   python -m venv .venv && .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. **Add files**
   - Drop your PDFs / `.txt` / `.md` files into `./data/`
3. **Run the app**
   ```bash
   streamlit run app.py
   ```
4. **Use it**
   - Click **Rebuild Index** once
   - Ask a question in the input box
   - Expand **Sources** to see the exact passages and files

## 📦 requirements.txt
```
streamlit
sentence-transformers
numpy
scikit-learn
nltk
pypdf
python-dotenv
```

## 🧠 How it works (short version)
1. **Ingest**: Files in `./data` are read. PDFs are extracted to text via `pypdf`, then split into sentences.
2. **Embed**: Sentences → 384‑dim vectors with `all-MiniLM-L6-v2` (CPU‑friendly).
3. **Search**: User query is embedded; we compute cosine similarity to find top‑k passages.
4. **Summarize**: A tiny TextRank‑ish routine composes a one‑paragraph answer from the top passages.
5. **Attribution**: Each result shows its source (and optional page number for PDFs).

## ⚙️ Options (optional niceties)
- **Page numbers in sources**: set `meta.append({"source": f.name, "page": i})` during PDF ingest.
- **Large PDFs**: if a file is huge, cap pages during ingest (e.g., `max_pages=300`) or split the PDF.
- **Batch encoding**: for very large corpora, encode in batches to keep RAM low.

## 🧹 .gitignore
```
# keep private data out of the repo
data/*.pdf
data/*.txt
data/*.md
index/*
```

## 🛡️ Legal & content note
Use your **own** documents or public‑domain texts for the repository. For private testing, you can use commercial books locally but **do not commit them** to Git.

## 🩹 Troubleshooting
- **Blank Streamlit page**: check terminal for errors, then refresh. Ensure `streamlit run app.py` is running.
- **Slow first run**: the model downloads once (~100 MB). After that, all offline.
- **Weird PDF text**: some PDFs have odd spacing/hyphenation. That’s normal with text extraction.
- **No results**: make sure files are inside `./data` and you clicked **Rebuild Index** after adding them.

## 🧰 Tech stack
- Python, Streamlit UI
- sentence-transformers (`all-MiniLM-L6-v2`), scikit‑learn cosine similarity
- pypdf for PDF text extraction
- NLTK sentence tokenization

## 📝 License
This project is provided under the MIT License (see `LICENSE`). The embedding model **all‑MiniLM‑L6‑v2** is Apache‑2.0 licensed.

## 🗣️ Portfolio blurb
**FlashFind — Local AI search for documents.** I built a CPU‑friendly semantic search app that indexes PDFs/notes and returns cited passages with a concise summary. Stack: sentence‑transformers + Streamlit + a lightweight TextRank summarizer. No cloud, no keys, zero cost.
