# Personal HR

Private, organization-scoped HR knowledge assistant. Employees ask policy questions from a Chrome side panel on registered portals; admins upload documents and manage organizations from a Next.js console. Answers are grounded in uploaded PDFs/DOCX, with optional employee memory and local Ollama models.

| Piece | Stack | Default URL |
| --- | --- | --- |
| API | FastAPI + CrewAI Flow + SQLite | `http://127.0.0.1:8000` |
| Web | Next.js 15 (App Router) | `http://127.0.0.1:3000` |
| Extension | Chrome MV3 | Load `extension/dist` |
| Models | Ollama (chat, planner, embeddings) | `http://127.0.0.1:11434` |

Architecture diagrams and a slide deck live under [`docs/`](docs/).

---

## Prerequisites

- **Python 3.12+**
- **Node.js 20+**
- **[Ollama](https://ollama.com/)** running locally
- **Chrome** (or Chromium) for the extension
- ~2 GB disk for embedding/reranker model downloads on first run

Pull the models used by the default config:

```bash
ollama pull qwen3.5:2b
ollama pull nomic-embed-text
```

---

## 1. Clone and layout

```text
PersonalHR/
├── server/          # FastAPI backend
├── web/             # Admin console + chat UI
├── extension/       # Chrome extension
├── docs/            # Architecture + presentation
└── Files/           # Optional sample policy PDFs
```

---

## 2. Backend (`server/`)

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment

Copy the example and fill in secrets (never commit `.env`):

```bash
cp .env.example .env
```

Create or edit `server/.env`:

```env
API_HOST=127.0.0.1
API_PORT=8000
WEB_ORIGIN=http://127.0.0.1:3000

ADMIN_EMAIL=admin@personalhr.local
ADMIN_PASSWORD_HASH=<bcrypt hash>

EXTENSION_KEY=<long random secret>

OLLAMA_BASE_URL=http://127.0.0.1:11434
CHAT_MODEL=qwen3.5:2b
PLANNER_MODEL=qwen3.5:2b
EMBED_MODEL=nomic-embed-text
EMBED_DIMENSIONS=768
RERANKER_MODEL=BAAI/bge-reranker-base

DATA_DIR=./data
FLOW_DATABASE=./data/flows.db
MAX_UPLOAD_MEGABYTES=15
```

Generate the admin password hash:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASSWORD', bcrypt.gensalt()).decode())"
```

Generate an extension key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Run the API

```bash
cd server
source .venv/bin/activate
python run.py
```

On first start the server loads the reranker and warms Ollama models. SQLite databases are created under `server/data/`.

Health / model status (after admin sign-in via the web app): `GET /admin/status`.

---

## 3. Web app (`web/`)

```bash
cd web
npm install
```

```bash
cp .env.example .env.local
```

Set in `web/.env.local`:

```env
API_BASE_URL=http://127.0.0.1:8000
EXTENSION_KEY=<same value as server EXTENSION_KEY>
```

Start the dev server:

```bash
npm run dev
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000).

**Typical admin path**

1. Sign in with `ADMIN_EMAIL` / your password  
2. Create an organization and set its **site host** (portal hostname)  
3. Upload policy PDFs/DOCX  
4. Tune assistant settings (greeting, answer style, portal/memory toggles)  
5. Use **Chat** or the extension on a registered host  

---

## 4. Chrome extension (`extension/`)

The build reads `EXTENSION_KEY` from `server/.env` and writes it into `extension/dist/`.

```bash
cd extension
npm install
npm run build
```

Load in Chrome:

1. Open `chrome://extensions`  
2. Enable **Developer mode**  
3. **Load unpacked** → select `extension/dist`  
4. Visit a registered organization portal while signed in — the **Ask Your HR** control appears  

The extension talks to the API with `X-Extension-Key` and only activates on hosts registered for an organization.

---

## 5. Quick verification

| Check | How |
| --- | --- |
| API up | `curl -s http://127.0.0.1:8000/docs` shows OpenAPI |
| Ollama | `ollama list` includes chat + embed models |
| Admin UI | Sign in at `http://127.0.0.1:3000` |
| Documents | Upload a PDF; status becomes `ready` |
| Chat | Ask a leave/policy question; expect citations |
| Extension | Panel on matching portal host |

Sample policies for demos are in `Files/` (gitignored locally; copy into the repo or upload from disk).

---

## Architecture

Full tech + implementation diagram (clients → API → CrewAI → RAG → memory → Ollama → SQLite):


| Layer | Tech | Implementation |
| --- | --- | --- |
| Clients | Chrome MV3 TypeScript, Next.js 15 / React 19 / Tailwind 4 | `extension/src`, `web/src` |
| API | FastAPI, Uvicorn, SSE | `server/app/routes`, `agent/streaming.py` |
| Orchestration | CrewAI Flow | `agent/flow.py`, `planner.py`, `agents.yaml`, `tools/` |
| RAG | pypdf, python-docx, sqlite-vec, FTS5, RRF, bge-reranker | `rag/ingestion`, `retrieval`, `generation.py` |
| Memory | SQLite employee facts + chat history | `memory/`, `chat/` |
| Models | Ollama `qwen3.5:2b`, `nomic-embed-text`; torch reranker | `llm/` |
| Persistence | SQLite `app.db`, `flows.db`, per-org vector DBs | `core/database.py`, `rag/vector_store.py` |

**Turn routes:** `GREETING` · `ANSWER` · `CLARIFY` · `OUT_OF_SCOPE` · `RESET` · `CACHED`

---

## Project map

| Path | Role |
| --- | --- |
| `server/app/agent/flow.py` | CrewAI conversation flow |
| `server/app/agent/planner.py` | Turn understanding / routing |
| `server/app/rag/` | Ingestion, retrieval, generation |
| `server/app/memory/` | Employee fact memory |
| `server/app/routes/` | HTTP API |
| `web/src/app/admin/` | Org, documents, models, activity |
| `web/src/app/chat/` | Browser chat UI |
| `extension/src/` | Content script + background worker |

---

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| Model warm-up warning | Start Ollama; `ollama pull` chat + embed models |
| Admin login fails | Regenerate `ADMIN_PASSWORD_HASH`; restart API |
| Extension build fails | Ensure `EXTENSION_KEY` is set in `server/.env` |
| Extension never shows | Site host must match portal hostname; user signed in |
| Empty / no-answer replies | Wait until document status is `ready`; ask a covered topic |
| Slow first chat | Reranker / embedding models download on first use |

---

## License

Internal / private project — confirm sharing policy with your team before publishing.
