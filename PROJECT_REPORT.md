Project Report: AI-Powered Recipe Assistant
Project Title: RAG-Powered Recipe Chatbot using Google Gemini

Technologies: Python, Streamlit, ChromaDB, Google Gemini API

Date: December 7, 2025

1. Introduction
The AI-Powered Recipe Assistant is an intelligent chatbot designed to help users discover cooking recipes and receive culinary advice. Unlike standard recipe search engines, this application uses Retrieval-Augmented Generation (RAG) to "read" a specific database of trusted recipes before answering. This ensures the AI provides accurate, specific instructions based on known data, rather than hallucinating ingredients.

2. Problem Statement
General AI models (like ChatGPT or standard Gemini) have vast knowledge but can sometimes invent recipes that do not exist or mix up ingredients. Additionally, users often struggle to find specific recipes within large, unorganized cookbooks or websites. There is a need for a tool that combines the conversational ability of AI with the accuracy of a dedicated recipe database.

3. Objectives
To build a user-friendly web interface using Streamlit.

To implement a Vector Database (ChromaDB) for storing and retrieving recipe data efficiently.

To integrate Google Gemini (LLM) to generate natural language responses.

To ensure the system runs locally on Windows with stability and rate-limit protection.

4. Technical Architecture
The system consists of three main components:

The Database (ChromaDB): Stores recipes not as text, but as "vector embeddings" (mathematical representations of meaning). This allows the system to understand that "spaghetti" is related to "pasta" even if the exact word isn't used.

The Brain (Google Gemini): A Large Language Model that takes the user's question and the relevant recipes found in the database to construct a helpful answer.

The Interface (Streamlit): A clean, web-based UI where users can chat with the bot and manage the database.

Workflow: User Query -> Search Vector DB -> Retrieve Best Matches -> Send to Gemini AI -> Generate Answer

5. Key Features
Semantic Search: Users can ask "What can I make with eggs?" and the system will find recipes like "Spaghetti Carbonara" or "French Toast" automatically.

Context-Aware Answers: The AI knows exactly which ingredients and steps belong to which recipe.

Rate-Limit Protection: Includes smart delays to prevent exceeding Google API free-tier limits.

Data Persistence: Recipes are saved locally, so the database does not need to be rebuilt every time the app restarts.

6. Implementation Details
Language: Python 3.13

Libraries:

streamlit: For the frontend UI.

chromadb: For vector storage.

google-generativeai: To connect to Gemini models.

Models Used:

Embedding: models/text-embedding-004

Generation: models/gemini-flash-latest (or gemini-1.5-flash)

7. Challenges & Solutions
Challenge: "Quota Exceeded" (429) errors from the Google API.

Solution: Implemented a 2-second sleep timer between database uploads to respect rate limits.

Challenge: Windows DLL/Connection errors.

Solution: Switched from local sentence-transformers to Google Embeddings and implemented connection caching (@st.cache_resource).

8. Conclusion
The project successfully demonstrates how RAG technology can create a "smart" cookbook. The chatbot provides accurate, context-based answers and offers a significant improvement over simple keyword searching. Future improvements could include adding image recognition for ingredients or deploying the app to the cloud.
