# AI Ordering Agent - Stage 2

## Features

- FastAPI backend
- OpenAI tool calling
- AI telecom ordering assistant
- Fake telecom offer database
- Quote calculation tools

---

## Setup

### 1. Create virtual environment

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

---

### 3. Create `.env`

Copy:

```bash
backend/.env.example
```

to:

```bash
backend/.env
```

Add your OpenAI API key.

---

### 4. Run server

```bash
cd backend
uvicorn main:app --reload
```

---

## Swagger Docs

Open:

http://127.0.0.1:8000/docs

---

## Example Requests

```json
{
  "message": "Find internet plans under 80 dollars"
}
```

```json
{
  "message": "Calculate quote for Internet Basic and Mobile Unlimited"
}
```

---

## Next Steps

Stage 3:
- conversation memory
- PostgreSQL
- workflow state
- frontend Angular UI
- RAG
- vector DB