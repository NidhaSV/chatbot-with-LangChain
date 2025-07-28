# Java Spark Frontend Service for Chatbot UI

This is a simple Java-based frontend service using the Spark framework. It provides a web-based chat UI that connects to the backend LLM + RAG service.

---

## Prerequisites

- Java 21 or higher
- Maven
- The backend service (`llm_connector_service`) must be running and accessible (default: http://localhost:8000)

---

## Build and Run

1. **Build the project:**
   ```bash
   mvn clean package
   ```

2. **Run the service:**
   ```bash
   mvn exec:java -D exec.mainClass="com.chatbot.Main"
   ```
   The server will start at [http://localhost:4567](http://localhost:4567).

---

## Access the Chat UI

- Open your browser and go to: [http://localhost:4567](http://localhost:4567)
- You will see the chatbot interface. Type your message and chat with the bot!

---

## Project Structure

- `src/main/resources/public/` — Static files (HTML, CSS, JS) for the chat UI
- `src/main/java/com/chatbot/` — Java source code (main entry: `Main.java`)

---

## Notes

- The frontend communicates with the backend at `http://localhost:8000` by default. If you need to change this, update the `BASE_URL` in `src/main/java/com/chatbot/Constant.java`.
- Make sure the backend is running before starting the frontend.
