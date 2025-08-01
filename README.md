### Internship project 1
# Chatbot with RAG

A simple chatbot enhanced with Retrieval Augmented Generation (RAG) using FAISS vector database and Anthropic's LLM API.  
This monorepo contains two main services:

- **llm_connector_service**: Python FastAPI backend for LLM + RAG
- **java_chat_client**: Java Spark-based frontend for the chat UI

Both services must be running for the chatbot to work.

---

## 1. Prerequisites

- **Python 3.8+** (for backend)
- **Java 21+** and **Maven** (for frontend)
- **Anthropic API Key** (for backend)

---

## 2. Setup and Run the Backend (llm_connector_service)

1. **Install dependencies:**
   ```bash
   cd llm_connector_service
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API Key:**
   - Create a `.env` file in `llm_connector_service/`:
     ```
     ANTHROPIC_API_KEY=<your_anthropic_api_key>
     ```

3. **Run the backend:**
   ```bash
   cd llm_connector_service
   python main.py
   ```
   The service will be available at [http://localhost:8000](http://localhost:8000).

---

## 3. Build and Run the Frontend (java_chat_client)

1. **Build the project:**
   ```bash
   cd java_chat_client
   mvn clean package
   ```

2. **Run the Java Spark service:**
   ```bash
   mvn exec:java -D exec.mainClass="com.chatbot.Main"
   ```
   This will start the frontend server at [http://localhost:4567](http://localhost:4567).

---

## 4. Load the Chat UI

- Open your browser and go to: [http://localhost:4567](http://localhost:4567)
- You should see the chatbot UI. Type your message and interact with the bot!

---

## 5. Project Structure

```
<Project Root>/
  ├── llm_connector_service/   # Python FastAPI backend (RAG pipeline)
  └── java_chat_client/        # Java Spark frontend (UI)
```

---

## 6. Notes

- The backend uses a demo CSV (`sap_defs.csv`) to populate the vector database.
- The frontend serves static files from `src/main/resources/public/` (including `index.html`, CSS, and JS).
- Make sure both services are running before using the chat UI.
