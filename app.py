"""
RAG-Powered Recipe Chatbot - Python 3.13 Compatible
No PyTorch required - Uses Google Embeddings API
"""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from typing import List, Dict
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    CHROMA_PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "recipes"
    TOP_K_RESULTS = 3
    GEMINI_MODEL = "gemini-1.5-flash"

# ============================================================================
# SAMPLE DATA
# ============================================================================

SAMPLE_RECIPES = [
    {
        "title": "Classic Chocolate Chip Cookies",
        "ingredients": "2 1/4 cups flour, 1 tsp baking soda, 1 cup butter, 3/4 cup sugar, 2 eggs, 2 cups chocolate chips",
        "instructions": "Preheat oven to 375°F. Mix dry ingredients. Beat butter and sugar. Add eggs. Mix in flour. Add chips. Bake 9-11 minutes.",
        "prep_time": "15 min", "cook_time": "11 min", "servings": "48 cookies", "category": "Dessert"
    },
    {
        "title": "Spaghetti Carbonara",
        "ingredients": "1 lb spaghetti, 6 oz pancetta, 4 eggs, 1 cup Parmesan, 2 cloves garlic",
        "instructions": "Cook pasta. Fry pancetta. Beat eggs with Parmesan. Toss hot pasta with pancetta, then egg mixture. Season with pepper.",
        "prep_time": "10 min", "cook_time": "15 min", "servings": "4", "category": "Main Course"
    },
    {
        "title": "Greek Salad",
        "ingredients": "4 tomatoes, 1 cucumber, 1 red onion, 1 bell pepper, 8 oz feta, 1/2 cup olives, olive oil, vinegar",
        "instructions": "Chop vegetables. Combine in bowl. Whisk oil, vinegar, oregano. Pour over vegetables. Top with feta.",
        "prep_time": "15 min", "cook_time": "0 min", "servings": "6", "category": "Salad"
    },
    {
        "title": "Chicken Tikka Masala",
        "ingredients": "1.5 lbs chicken, 1 cup yogurt, tikka spice, onion, garlic, ginger, tomato sauce, cream",
        "instructions": "Marinate chicken. Grill chicken. Sauté aromatics. Add sauce and cream. Add chicken. Simmer.",
        "prep_time": "2 hrs", "cook_time": "30 min", "servings": "6", "category": "Main Course"
    },
    {
        "title": "Banana Bread",
        "ingredients": "3 bananas, 1/3 cup butter, 1 tsp baking soda, 3/4 cup sugar, 1 egg, 1.5 cups flour",
        "instructions": "Mash bananas. Mix in butter and baking soda. Add sugar, egg, vanilla. Mix in flour. Bake 60 min at 350°F.",
        "prep_time": "10 min", "cook_time": "60 min", "servings": "8 slices", "category": "Dessert"
    },
    {
        "title": "Caesar Salad",
        "ingredients": "Romaine lettuce, Caesar dressing, croutons, Parmesan cheese, lemon",
        "instructions": "Chop lettuce. Toss with dressing and croutons. Top with Parmesan and lemon juice.",
        "prep_time": "10 min", "cook_time": "0 min", "servings": "4", "category": "Salad"
    },
    {
        "title": "Beef Tacos",
        "ingredients": "1 lb ground beef, taco seasoning, taco shells, lettuce, tomato, cheese, sour cream",
        "instructions": "Brown beef. Add seasoning. Warm shells. Fill with beef and toppings.",
        "prep_time": "10 min", "cook_time": "15 min", "servings": "4", "category": "Main Course"
    },
    {
        "title": "French Toast",
        "ingredients": "4 eggs, 1/4 cup milk, vanilla, cinnamon, 8 bread slices, butter, maple syrup",
        "instructions": "Beat eggs, milk, vanilla, cinnamon. Dip bread. Cook in butter until golden. Serve with syrup.",
        "prep_time": "5 min", "cook_time": "15 min", "servings": "4", "category": "Breakfast"
    },
    {
        "title": "Vegetable Stir Fry",
        "ingredients": "Broccoli, bell pepper, snap peas, carrot, garlic, ginger, soy sauce, sesame oil",
        "instructions": "Cut vegetables. Heat oil. Add garlic and ginger. Stir fry vegetables. Add sauce.",
        "prep_time": "15 min", "cook_time": "10 min", "servings": "4", "category": "Main Course"
    },
    {
        "title": "Blueberry Muffins",
        "ingredients": "2 cups flour, 3/4 cup sugar, baking powder, oil, egg, milk, vanilla, blueberries",
        "instructions": "Mix dry ingredients. Whisk wet ingredients. Combine. Fold in blueberries. Bake 20-25 min at 400°F.",
        "prep_time": "10 min", "cook_time": "25 min", "servings": "12 muffins", "category": "Breakfast"
    },
    {
        "title": "Margherita Pizza",
        "ingredients": "Pizza dough, tomato sauce, fresh mozzarella, basil, olive oil",
        "instructions": "Roll dough. Spread sauce. Add mozzarella. Bake at 475°F for 12-15 min. Top with basil.",
        "prep_time": "15 min", "cook_time": "15 min", "servings": "2-4", "category": "Main Course"
    },
    {
        "title": "Chicken Noodle Soup",
        "ingredients": "Chicken breast, broth, carrots, celery, onion, garlic, egg noodles, herbs",
        "instructions": "Simmer chicken in broth. Shred chicken. Sauté vegetables. Add to broth. Add noodles and chicken.",
        "prep_time": "15 min", "cook_time": "45 min", "servings": "6", "category": "Soup"
    },
    {
        "title": "Guacamole",
        "ingredients": "3 avocados, lime juice, salt, onion, cilantro, tomatoes, jalapeño, garlic",
        "instructions": "Mash avocados. Mix in lime and salt. Add remaining ingredients. Serve with chips.",
        "prep_time": "10 min", "cook_time": "0 min", "servings": "6", "category": "Appetizer"
    },
    {
        "title": "Beef Stroganoff",
        "ingredients": "Beef sirloin, mushrooms, onion, garlic, flour, beef broth, sour cream, egg noodles",
        "instructions": "Sauté beef. Cook mushrooms and onions. Add flour and broth. Add sour cream. Serve over noodles.",
        "prep_time": "15 min", "cook_time": "30 min", "servings": "4", "category": "Main Course"
    },
    {
        "title": "Caprese Salad",
        "ingredients": "Tomatoes, fresh mozzarella, basil, olive oil, balsamic vinegar",
        "instructions": "Slice tomatoes and mozzarella. Arrange with basil. Drizzle with oil and vinegar.",
        "prep_time": "10 min", "cook_time": "0 min", "servings": "4", "category": "Salad"
    }
]

# ============================================================================
# VECTOR DATABASE MANAGER
# ============================================================================

class VectorDBManager:
    """Manages ChromaDB with Google Embeddings"""
    
    def __init__(self, api_key: str):
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name="models/embedding-001"
        )
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        self.collection = None
        
    def initialize_collection(self):
        """Initialize or get existing collection"""
        try:
            self.collection = self.client.get_collection(
                name=Config.COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=Config.COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_recipes(self, recipes: List[Dict]):
        """Add recipes to vector database"""
        if not recipes:
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, recipe in enumerate(recipes):
            doc_text = f"""Title: {recipe['title']}
Category: {recipe['category']}
Ingredients: {recipe['ingredients']}
Instructions: {recipe['instructions']}
Prep: {recipe['prep_time']} | Cook: {recipe['cook_time']} | Servings: {recipe['servings']}"""
            
            documents.append(doc_text)
            metadatas.append(recipe)
            ids.append(f"recipe_{idx}")
        
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    def search(self, query: str, top_k: int = Config.TOP_K_RESULTS) -> List[Dict]:
        """Search for relevant recipes"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results['metadatas'][0] if results['metadatas'] else []
    
    def get_all_recipes(self) -> List[Dict]:
        """Get all recipes from database"""
        results = self.collection.get()
        return results['metadatas'] if results['metadatas'] else []

# ============================================================================
# RAG CHATBOT
# ============================================================================

class RAGChatbot:
    """RAG-powered chatbot using Google Gemini"""
    
    def __init__(self, api_key: str, vector_db: VectorDBManager):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.vector_db = vector_db
        self.chat = self.model.start_chat(history=[])
    
    def generate_response(self, user_query: str) -> str:
        """Generate response using RAG pipeline"""
        try:
            # Retrieve relevant recipes
            recipes = self.vector_db.search(user_query)
            
            # Format context
            if recipes:
                context = "\n\n".join([
                    f"Recipe: {r['title']}\nCategory: {r['category']}\n"
                    f"Ingredients: {r['ingredients']}\nInstructions: {r['instructions']}\n"
                    f"Time: {r['prep_time']} prep, {r['cook_time']} cook | Serves: {r['servings']}"
                    for r in recipes
                ])
            else:
                context = "No specific recipes found."
            
            # Create prompt
            prompt = f"""Based on these recipes:

{context}

User question: {user_query}

Provide a helpful, friendly response about cooking and recipes."""
            
            # Generate response
            response = self.chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.chat = self.model.start_chat(history=[])

# ============================================================================
# STREAMLIT UI
# ============================================================================

def initialize_session_state():
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

def setup_database(api_key: str):
    with st.spinner("Setting up database..."):
        db_manager = VectorDBManager(api_key)
        db_manager.initialize_collection()
        
        if db_manager.collection.count() == 0:
            st.info("Loading sample recipes...")
            db_manager.add_recipes(SAMPLE_RECIPES)
            st.success(f"Loaded {len(SAMPLE_RECIPES)} recipes!")
        
        return db_manager

def main():
    st.set_page_config(page_title="Recipe RAG Chatbot", page_icon="🍳", layout="wide")
    
    st.markdown("""
        <style>
        .main-header {text-align: center; color: #FF6B6B; padding: 1rem 0;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>🍳 Recipe RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Powered by Google Gemini • Python 3.13 Compatible ✅</p>", unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=Config.GOOGLE_API_KEY,
            help="Get your key at https://makersuite.google.com/app/apikey"
        )
        
        model_option = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            help="Flash = faster, Pro = better quality"
        )
        Config.GEMINI_MODEL = model_option
        
        if st.button("Initialize System", type="primary"):
            if not api_key:
                st.error("Please provide an API key")
            else:
                try:
                    st.session_state.db_manager = setup_database(api_key)
                    st.session_state.chatbot = RAGChatbot(api_key, st.session_state.db_manager)
                    st.session_state.initialized = True
                    st.success("✅ System initialized!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.divider()
        
        if st.session_state.db_manager:
            st.header("📊 Database Stats")
            st.metric("Total Recipes", st.session_state.db_manager.collection.count())
            
            if st.button("View All Recipes"):
                recipes = st.session_state.db_manager.get_all_recipes()
                with st.expander("Recipe List", expanded=True):
                    for recipe in recipes:
                        st.write(f"- {recipe['title']} ({recipe['category']})")
        
        st.divider()
        
        if st.button("🔄 Reset Conversation"):
            if st.session_state.chatbot:
                st.session_state.chatbot.reset_conversation()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.markdown("### 💡 Try These")
        st.markdown("""
        - What desserts do you have?
        - How do I make carbonara?
        - Quick breakfast ideas?
        - Recipes with chicken?
        """)
        
        st.divider()
        st.info("✅ No PyTorch • Python 3.13 Compatible")
    
    # Main chat area
    if not st.session_state.initialized:
        st.info("👈 Initialize the system in the sidebar to get started!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Features:**
            - 🔍 Semantic recipe search
            - 💬 Conversational AI
            - ⚡ Fast & lightweight
            - ✅ Python 3.13 ready
            """)
        with col2:
            st.markdown("""
            **Tech:**
            - ChromaDB vectors
            - Google Embeddings
            - Gemini AI
            - No PyTorch needed!
            """)
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about recipes..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.generate_response(prompt)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()