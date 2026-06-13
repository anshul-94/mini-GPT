# 🧠 Memory-Driven Conversational AI

A lightweight conversational AI system designed to maintain context across interactions and provide more personalized responses through memory-aware conversation handling.

The chatbot combines short-term conversational context with intelligent memory management to create a more natural and human-like chat experience.

---

## 🚀 Project Overview

Traditional chatbots often lose context as conversations grow longer.

This project introduces a memory-driven approach where the assistant can:

* Remember recent conversations
* Maintain conversational context
* Generate personalized responses
* Improve continuity across interactions
* Simulate more natural human-like conversations

The system is designed as a foundation for advanced AI assistants and future agent-based workflows.

---

## ✨ Features

### Short-Term Memory (STM)

* Maintains recent conversation history
* Preserves context during active chats
* Supports multi-turn interactions

### Context-Aware Responses

* Understands ongoing discussions
* Produces more relevant answers
* Reduces repetitive responses

### Session-Based Conversations

* Separate conversation sessions
* Independent chat histories
* Improved user experience

### Responsive User Interface

* Mobile-friendly design
* Modern chatbot interface
* Real-time messaging experience

### OpenRouter Integration

* Supports modern LLMs
* Easy model switching
* Flexible AI backend

---

## 🏗️ System Workflow

```text
User Message
      ↓
Conversation Context Retrieval
      ↓
Short-Term Memory Processing
      ↓
Prompt Construction
      ↓
LLM Response Generation
      ↓
Response Delivery
      ↓
Memory Update
```

---

## 🛠️ Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* FastAPI
* Uvicorn

### AI Layer

* OpenRouter API
* LangChain
* Prompt Engineering
* Conversational AI

### Database (Planned / Future Scope)

* PostgreSQL

### AI Concepts

* NLP
* Context Management
* Memory Systems
* Retrieval-Augmented Generation (RAG)

---

## 📂 Project Structure

```text
project/
│
├── main.py
├── chat.py
├── index.html
├── requirements.txt
├── README.md
└── .env
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/anshul-94/mini-GPT.git
cd mini-GPT
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Mac/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Optional:

```env
OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct
```

---

## ▶️ Running the Project

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Server will start at:

```text
http://127.0.0.1:8000
```

Open the URL in your browser and start chatting.

---

## 🔮 Future Improvements

* Long-Term Memory
* Vector Database Integration
* User Profiles
* AI Agents
* Web Search Tools
* File Analysis
* Memory Summarization
* Multi-User Support

---
## 📌 Project Goal

Build a conversational AI system that not only responds to messages but also remembers context, understands conversations, and provides a more intelligent user experience.
