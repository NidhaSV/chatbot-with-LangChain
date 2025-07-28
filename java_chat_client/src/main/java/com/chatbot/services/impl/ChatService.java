package com.chatbot.services.impl;

public interface ChatService {

    /**
     * Sends the user's message to the backend and returns the bot response.
     */
    String sendMessage(String message);

}