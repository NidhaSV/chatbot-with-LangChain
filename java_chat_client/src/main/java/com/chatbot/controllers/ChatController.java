package com.chatbot.controllers;
import com.chatbot.services.impl.ChatService;
import com.fasterxml.jackson.core.type.TypeReference;

import static spark.Spark.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.HashMap;
import java.util.Map;
import com.fasterxml.jackson.core.type.TypeReference;

public class ChatController {
    private final ChatService chatService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public ChatController(ChatService chatService) {
        this.chatService = chatService;
        setupRoutes();
    }

    private void setupRoutes() {
        // POST /api/chat - handle chat messages
        post("/api/chat", (req, res) -> {
            res.type("application/json");
            try {
                Map<String, String> body = objectMapper.readValue(
                    req.body(),
                    new TypeReference<Map<String, String>>() {}
                );
                String prompt = body.getOrDefault("prompt", "");
                if (prompt.isEmpty()) {
                    res.status(400);
                    return objectMapper.writeValueAsString(Map.of("error", "Prompt cannot be empty."));
                }
                String response = chatService.sendMessage(prompt);
                return objectMapper.writeValueAsString(Map.of("response", response));
            } catch (Exception e) {
                res.status(500);
                return objectMapper.writeValueAsString(Map.of("error", "Internal server error."));
            }
        });
    }
} 