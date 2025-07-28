package com.chatbot;
import static spark.Spark.*;

import com.chatbot.controllers.ChatController;
import com.chatbot.services.impl.ChatService;
import com.chatbot.services.impl.RagChatService;

public class Main {
    public static void main(String[] args) {
        port(4567); // Default SparkJava port

        // Serve static files (index.html, JS, CSS) from /public
        staticFiles.location("/public");

        // Optionally, add a health check endpoint
        get("/health", (req, res) -> "OK");

    
        ChatService chatService = new RagChatService();
        new ChatController(chatService);
    }
}