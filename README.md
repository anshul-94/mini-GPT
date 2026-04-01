# 🧠 Memory-Driven Mini ChatGPT

A conversational AI chatbot that remembers user interactions, understands context, and delivers personalized responses over time.

---

## 🚀 Project Overview

This project is a **memory-driven AI chatbot** inspired by ChatGPT.

### 💡 Idea
The chatbot can:
- Remember previous conversations  
- Understand context  
- Give personalized replies  
- Improve responses over time  

---

## 🎯 Objectives

- Build human-like conversational AI  
- Implement Short-Term Memory (STM)  
- Implement Long-Term Memory (LTM)  
- Use past data for better responses  
- Prepare system for future AI agents & tools  

---

## 🧠 Core Concepts

### 🔹 Short-Term Memory (STM)
- Stores last 5–10 messages  
- Maintains conversation flow  

Example:  
User: My name is Anshul  
Bot remembers this for next messages  

---

### 🔹 Long-Term Memory (LTM)
- Stores important user info  
- Preferences, interests, goals  

Example:  
User: I like gaming laptops  
Bot stores this for future replies  

---

### 🔹 Memory Retrieval
- Searches past data  
- Sends relevant memory to AI before response  

---

## ⚙️ System Architecture

User Input  
↓  
Short-Term Memory (Recent Chat)  
↓  
Long-Term Memory (Search Past Data)  
↓  
Combine Context  
↓  
AI Model (LLM)  
↓  
Generate Response  
↓  
Store Important Info  

---

## ✨ Key Features

- Context Awareness  
- Personalized Responses  
- Smart Memory Storage  
- Works Across Sessions  
- Memory Retrieval System  

---

## 🔥 Advanced Features

- Memory Summarization  
- Forget Option (User Controlled Memory)  
- Multi-user Support  
- Security Handling  

---

## 🧩 Prompt Design

You are a helpful AI assistant.

User Info:  
{long_term_memory}

Recent Chat:  
{short_term_memory}

User Question:  
{query}

---

## 🛠️ Tech Stack

### Backend
- FastAPI  

### AI Layer
- LangChain  
- LangGraph (future)  
- OpenAI API / OpenRouter  

### Database
- PostgreSQL (chat history)  
- FAISS / ChromaDB (memory storage)  

### Frontend
- HTML / CSS / JavaScript (current)  
- React / Next.js (future upgrade)  

---

## 👥 Team Roles

- Memory System → Store & retrieve memory, vector DB  
- AI Logic → Prompt design, response generation  
- Backend → APIs, database  
- Frontend → UI & UX  

---

## 🔄 Workflow

1. User sends message  
2. Fetch recent chat (STM)  
3. Retrieve relevant memory (LTM)  
4. Send context to AI  
5. Generate response  
6. Store useful data  

---

## 🌍 Future Scope

- Web Search Integration  
- Image Generation  
- File Upload & Analysis  
- Full AI Agents System  

---

## 💪 Why This Project is Strong

- Real-world AI system  
- Uses modern concepts:
  - NLP  
  - RAG  
  - Memory Systems  
- Scalable & extendable  

---

## 🎯 Final Outcome

This chatbot will:
- Remember users  
- Understand context  
- Give smarter replies  
- Improve over time  

---

## 🏁 Conclusion

This is not just a chatbot.  
It is a smart AI system that learns about the user.

---

## ⚡ End Goal

Build AI that remembers + understands + improves
