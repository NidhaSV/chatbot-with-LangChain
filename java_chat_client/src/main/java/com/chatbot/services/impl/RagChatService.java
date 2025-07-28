package com.chatbot.services.impl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.chatbot.Constant;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

public class RagChatService implements ChatService {

    private static final String LLM_QUERY_URL = Constant.BASE_URL + "/llm/query";

    private static final Logger logger = LoggerFactory.getLogger(RagChatService.class);

    private final OkHttpClient client = new OkHttpClient();

    private final ObjectMapper objectMapper = new ObjectMapper();


    /**
     * Sends the user's prompt to the FastAPI backend enhanced with RAG and returns
     * the LLM response. Handles both streaming and non-streaming responses.
     */
    @Override
    public String sendMessage(String prompt) {
        try {
            RequestBody body = RequestBody.create(
                objectMapper.writeValueAsString(new PromptRequest(prompt)),
                MediaType.parse("application/json")
            );
            Request request = new Request.Builder().url(LLM_QUERY_URL).post(body).build();
            Response response = client.newCall(request).execute();
            if (!response.isSuccessful()) {
                return "[Error: Backend returned status " + response.code() + "]";
            }
            // Try to handle streaming (text/event-stream) or JSON
            String contentType = response.header("Content-Type", "");
            logger.info("LLM API response Content-Type: {}", contentType);
            if (contentType.contains("text/event-stream")) {
                // Read the stream and concatenate chunks
                StringBuilder messageBuilder = new StringBuilder();
                try (ResponseBody respBody = response.body()) {
                    if (respBody != null) {
                        String respBodyString = respBody.string();
                        logger.debug("Response Body: {}", respBodyString);
                        for (String line : respBodyString.split("\n")) {
                            if (!line.trim().isEmpty()) {
                                String delta = getTextDeltaFromStreamBody(line);
                                messageBuilder.append(delta);
                            }
                        }
                    }
                }
                return messageBuilder.toString().trim();
            } else {
                // Assume JSON `{"response": ...}`
                String respStr = response.body().string();
                JsonNode node = objectMapper.readTree(respStr);
                if (node.has("response")) {
                    return node.get("response").asText();
                } else if (node.has("detail")) {
                    return node.get("detail").asText();
                } else {
                    return respStr;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "[Network or parsing error: " + e.getMessage() + "]";
        }
    }

    private String getTextDeltaFromStreamBody(String line) {
        if (line.startsWith("data:")) {
            String jsonString = line.substring(5).trim(); // remove "data:" prefix

            // Try parsing JSON and extracting text if it's a content_block_delta
            try {
                JsonNode root = objectMapper.readTree(jsonString);
                if (root.has("type") && "content_block_delta".equals(root.get("type").asText())) {
                    JsonNode deltaNode = root.path("delta");
                    if (deltaNode.has("text")) {
                        return deltaNode.get("text").asText();
                    }
                }
            } catch (JsonProcessingException ex) {
                logger.error("Error parsing JSON: {}", ex.getMessage(), ex);
                // Skip malformed JSON or non-matching structures
                logger.error("Skipped invalid JSON line: {}", jsonString);
            }
        }
        return "";
    }


    // Helper class for prompt JSON
    static class PromptRequest {

        public String prompt;


        public PromptRequest(String prompt) {
            this.prompt = prompt;
        }

    }

}