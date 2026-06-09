# 🧠 AI Chatbot Backend

A lightweight, high-performance, and robust backend for an AI Chatbot. Built with Python, FastAPI, and Uvicorn, it integrates with the OpenRouter API to fetch responses using the `openai/gpt-oss-20b:free` model (or any model of your choice).

This backend is designed to work seamlessly with the pure HTML/CSS/JS frontend included in `index.html`.

---

## 🛠️ Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast (high-performance) web framework for building APIs.
- **HTTP Client:** [Requests](https://requests.readthedocs.io/) - Synchronous HTTP library for Python to connect to the OpenRouter API.
- **ASGI Server:** [Uvicorn](https://www.uvicorn.org/) - Lightning-fast ASGI server implementation.
- **Configuration:** [python-dotenv](https://github.com/theofidry/django-dotenv) - Reads key-value pairs from a `.env` file and sets them as environment variables.

---

## 📂 Project Structure

```text
project/
│
├── main.py              # Main FastAPI application with routes and error handling
├── requirements.txt     # Python packages and version requirements
├── .env                 # Environment secrets (API Keys, config) - ignored by git
├── .env.example         # Template for environment secrets
└── README.md            # Installation and setup documentation (this file)
```

---

## ⚙️ Installation & Setup

Follow these steps to set up and run the backend locally:

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You can check your Python version with:
```bash
python --version
```

### 2. Create a Virtual Environment (Recommended)
It is recommended to run the project inside a virtual environment to avoid package conflicts:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (cmd):
# venv\Scripts\activate.bat
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install all required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configuration
1. Copy the template `.env.example` file to create your `.env` configuration file:
   ```bash
   cp .env.example .env
   ```
2. Open the newly created `.env` file and replace `YOUR_OPENROUTER_API_KEY_HERE` with your actual OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-...
   ```
   > 💡 You can obtain your API key from [OpenRouter Keys Dashboard](https://openrouter.ai/keys).

### 5. Running the Backend Server
Start the Uvicorn development server with hot-reload enabled:
```bash
uvicorn main:app --reload
```
The server will start at **`http://127.0.0.1:8000`**. You can verify the health by navigating to `http://127.0.0.1:8000/health` in your browser.

---

## 🔌 API Documentation

### **POST** `/chat`
Sends a prompt message to the backend, which forwards it to OpenRouter and returns the AI reply.

* **Headers:**
  `Content-Type: application/json`

* **Request Body:**
  ```json
  {
    "message": "Hello, how are you?",
    "session_id": "abc123"
  }
  ```

* **Successful Response (200 OK):**
  ```json
  {
    "reply": "Hello! I am doing well, thank you. How can I help you today?"
  }
  ```

* **Error Response Example (400 Bad Request):**
  ```json
  {
    "reply": "Error: Message cannot be empty."
  }
  ```

* **CORS Configured:**
  Cross-Origin Resource Sharing (CORS) is enabled (`allow_origins=["*"]`) so that `index.html` can call the endpoint directly when run locally from the browser.

---

## 🤖 Changing the AI Model
To switch to a different AI model (including newer or paid models):
1. Open your `.env` file.
2. Edit the `OPENROUTER_MODEL` variable. For example, to use Meta Llama 3:
   ```env
   OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct:free
   ```
3. Save the file. The server will automatically reload and use the new model on subsequent chat requests!

---

## 🎨 Testing the Full Application

1. Make sure your FastAPI backend is running on `http://127.0.0.1:8000`.
2. Open the `index.html` file in any modern web browser.
3. Type a message in the input box at the bottom and press **Enter** or click the send button.
4. You will see a loading indicator `"AI is thinking..."` and receive the streaming response shortly.
