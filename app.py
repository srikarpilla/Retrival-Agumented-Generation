"""
RAG-Powered Recipe Chatbot (Stable Windows Version)
Google Gemini 2.0 + Google Embeddings + Streamlit Secrets Support
"""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import os
from typing import List, Dict
import time

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
class Config:
    # API Key: First Try Streamlit Secrets, fallback to environment
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

    CHROMA_PERSIST_DIR = "./chroma_db_stable"
    COLLECTION_NAME = "recipes_v6"

    EMBEDDING_MODEL = "models/text-embedding-004"
    GEMINI_MODEL = "models/gemini-2.0-flash-exp"


# ============================================================================
# 2. DATA GENERATOR
# ============================================================================
class RecipeScraper:
    @staticmethod
    def scrape_sample_recipes() -> List[Dict]:
        return [
            {
                "title": "Classic Chocolate Chip Cookies",
                "ingredients": "2 1/4 cups flour, 1 tsp baking soda, 1 cup butter, 3/4 cup sugar, 3/4 cup brown sugar, 2 eggs, 2 tsp vanilla, 2 cups chocolate chips",
                "instructions": "Mix flour and soda. Beat butter and sugars. Add eggs and vanilla. Stir in flour then chips. Bake at 375°F for 9-11 mins.",
                "prep_time": "15 m", "cook_time": "11 m", "servings": "48 cookies", "category": "Dessert"
            },
            {
                "title": "Spaghetti Carbonara",
                "ingredients": "1 lb spaghetti, 6 oz pancetta, 4 eggs, 1 cup Parmesan, garlic, pepper",
                "instructions": "Cook pasta. Fry pancetta. Beat eggs with cheese. Toss hot pasta with pancetta and egg mixture quickly to create creamy sauce.",
                "prep_time": "10 m", "cook_time": "15 m", "servings": "4 servings", "category": "Main Course"
            },
            {
                "title": "Chicken Tikka Masala",
                "ingredients": "Chicken breast, yogurt, tikka spices, onion, garlic, tomato sauce, heavy cream, butter",
                "instructions": "Marinate chicken. Grill it. Simmer onion, spices, and tomato sauce. Add cream and chicken. Simmer 10 mins.",
                "prep_time": "2 hr", "cook_time": "30 m", "servings": "6 servings", "category": "Main Course"
            },
            {
                "title": "Greek Salad",
                "ingredients": "Tomatoes, cucumber, red onion, feta cheese, olives, olive oil, oregano",
                "instructions": "Chop veggies. Mix with olives. Whisk oil and oregano. Toss everything together. Top with feta.",
                "prep_time": "15 m", "cook_time": "0 m", "servings": "6 servings", "category": "Salad"
            },
            {
                "title": "Banana Bread",
                "ingredients": "3 bananas, 1/3 cup melted butter, 1 tsp baking soda, 3/4 cup sugar, 1 egg, 1.5 cups flour",
                "instructions": "Mash bananas. Mix in butter, soda, sugar, egg. Fold in flour. Bake at 350°F for 60 mins.",
                "prep_time": "10 m", "cook_time": "60 m", "servings": "8 slices", "category": "Dessert"
            }
        ]


# ============================================================================
# 3. VECTOR DATABASE MANAGER
# ============================================================================
class VectorDBManager:
    def __init__(self, api_key: str):
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name=Config.EMBEDDING_MODEL
        )

        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        self.collection = None
        
    def initialize_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=Config.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:num_threads": 1}
            )
        except Exception as e:
            st.error(f"Database Error: {e}")
    
    def add_recipes(self, recipes: List[Dict]):
        if not recipes:
            return
        
        if self.collection.count() > 0:
            return

        progress = st.progress(0, text="Adding recipes...")
        total = len(recipes)

        for idx, recipe in enumerate(recipes):
            doc = (
                f"Title: {recipe['title']} | "
                f"Ingredients: {recipe['ingredients']} | "
                f"Instructions: {recipe['instructions']}"
            )

            self.collection.add(
                documents=[doc],
                metadatas=[recipe],
                ids=[f"recipe_{idx}"]
            )

            time.sleep(1.5)
            progress.progress((idx + 1) / total, text=f"Added {recipe['title']}")

        progress.empty()
        st.success(f"Loaded {total} recipes!")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.collection:
            return []
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results["metadatas"][0] if results["metadatas"] else []


# ============================================================================
# 4. RAG CHATBOT
# ============================================================================
class RAGChatbot:
    def __init__(self, api_key: str, vector_db: VectorDBManager):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.vector_db = vector_db
        self.chat = self.model.start_chat(history=[])
    
    def generate_response(self, user_query: str) -> str:
        retrieved = self.vector_db.search(user_query)
        
        if retrieved:
            ctx = "\n\n".join([
                f"Name: {r['title']}\nIngredients: {r['ingredients']}\nInstructions: {r['instructions']}"
                for r in retrieved
            ])
            prompt = f"""
You are a chef assistant. Use these recipes to answer the user request.

=== CONTEXT ===
{ctx}

=== USER ===
{user_query}

Use recipe content if relevant.
"""
        else:
            prompt = f"User asked: {user_query}. Provide a helpful cooking answer."

        try:
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            return f"AI Error: {e}"


# ============================================================================
# 5. GLOBAL CHROMA SINGLETON (Prevents DB Disconnect on Windows)
# ============================================================================

_GLOBAL_DB = None

def get_db_connection(api_key: str):
    global _GLOBAL_DB
    if _GLOBAL_DB is None:
        try:
            _GLOBAL_DB = VectorDBManager(api_key)
            _GLOBAL_DB.initialize_collection()
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    return _GLOBAL_DB


# ============================================================================
# 6. MAIN APP UI
# ============================================================================
def main():
    st.set_page_config(page_title="Chef Bot", page_icon="🍳")
    st.title("🍳 AI Recipe Assistant (Gemini 2.0)")

    # Get API key directly from secrets
    api_key = Config.GOOGLE_API_KEY

    if not api_key:
        st.error("❌ No Google API Key found in Streamlit Secrets or environment variables.")
        st.info("Add a secrets.toml file:\n\nGOOGLE_API_KEY = \"your_key_here\"")
        return

    # Initialize DB + chatbot
    db = get_db_connection(api_key)

    if db is None:
        st.error("Database unavailable.")
        return

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot(api_key, db)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("Database")
        if db.collection.count() == 0:
            if st.button("Load Sample Recipes"):
                db.add_recipes(RecipeScraper.scrape_sample_recipes())
                st.rerun()
        else:
            st.success(f"{db.collection.count()} recipes loaded")

    # Chat UI
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("What would you like to cook?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Thinking..."):
            reply = st.session_state.chatbot.generate_response(prompt)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)


if __name__ == "__main__":
    main()

