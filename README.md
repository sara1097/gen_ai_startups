# AI Startup Idea Generator

AI-Powered System That analyze The Egyptian market problems and generate a new startup ideas inspired by an foreign ones.
## Features
- Intent detection
- Problem extraction
- Reasoning router
- RAG-based idea retrieval
- AI startup idea generation
- FastAPI backend

## Project Structure
```
app/
│
├── src/
| ├── chat_Schemas
|   └── schemas.py
│ ├── engine/
│ | ├── core/
| |    ├── intent_classification.py
| |    └── reasoning_router.py
│ │ └── rag/
| |    └── retriver.py
│ 
│ ├── llm/
│ │ ├── base.py
│ │ └── groq_provider.py
│ │
│ ├── prompt_Engineering/
│ │ ├── templates.py
│ │ └── few_shot.py
│
│
├── main.py
```
## Installation
Clone the repository
```
git clone https://github.com/username/startup-ai-generator.git
```
Install dependencies
```
pip install -r requirments.txt
```
Run the server
```
uvicorn app.main:app --reload
```

## Technologies

- FastAPI
- Groq LLM
- Python
- RAG Architecture
- Prompt Engineering

## Future Improvements

- Improve intent classification
- Multi-agent reasoning