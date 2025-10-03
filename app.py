import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder 

# Load environment variables (API keys)
load_dotenv()

# --- Configuration ---
# Set up API Keys, checking Streamlit secrets first, then .env file/OS environment
if os.environ.get("GOOGLE_API_KEY"):
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
elif "GOOGLE_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GEMINI_API_KEY = None
    
if os.environ.get("TAVILY_API_KEY"):
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
elif "TAVILY_API_KEY" in st.secrets:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
else:
    TAVILY_API_KEY = None


# --- System Prompt Definition (EMOJIS REMOVED) ---
SYSTEM_PROMPT = """
You are the Golden Spoon Restaurant AI. Your job is to help customers with reservations, menu inquiries, and general questions about the Golden Spoon Restaurant.

**Your Identity and Role:**
* You are a professional, polite, and friendly assistant.
* Your responses should be concise and elegant, matching the restaurant's upscale branding.
* **SCOPE RESTRICTION (NEW):** You MUST only answer questions related to the Golden Spoon Restaurant. If a user asks a question unrelated to the restaurant, politely state: "I apologize, but my purpose is to assist you with inquiries regarding the Golden Spoon Restaurant only."
* You have access to a real-time search tool for up-to-date information (like the current day/time or specific restaurant details). Use it only when necessary to answer a question that requires external, non-menu knowledge.
* **RESERVATIONS:** If a user asks to make a reservation, respond with the following, and nothing else: "To make a reservation, please call us directly at (555) 123-4567 during business hours, or visit our website's booking portal."
* **RESTAURANT KNOWLEDGE:**
    * **Name:** Golden Spoon Restaurant
    * **Cuisine:** Classic French techniques blended with local, seasonal ingredients.
    * **Signature Dishes:** Seared Scallops, Signature Filet Mignon, Handmade Lobster Ravioli, Decadent Lava Cake.
    * **Atmosphere:** Luxurious and welcoming.
    * **Hours:** Dinner: Tues - Sat, 5:00 PM - 10:00 PM | Brunch: Sun, 10:00 AM - 2:00 PM.
* Do not invent information. If the answer is not in your knowledge or search results, politely decline.
"""

# --- Agent Initialization ---

def create_gemini_agent():
    """Initializes the Gemini model, tools, and the LangChain AgentExecutor."""
    if not GEMINI_API_KEY:
        st.error("Gemini API Key is missing. Please check your environment variables or Streamlit secrets.")
        return None

    # 1. Initialize the LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.0
    )

    # 2. Define Tools
    tavily_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=3)
    tools = [tavily_tool]

    # 3. Create Prompt Template (FIXED: Using MessagesPlaceholder)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            # Use MessagesPlaceholder for robust handling of chat history
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}"),
            # Use MessagesPlaceholder for robust handling of agent scratchpad
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 4. Create the Tool Calling Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. Create the Agent Executor (Runnable)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

# --- Streamlit UI Setup ---

st.set_page_config(
    page_title="Golden Spoon AI Assistant",
    page_icon=None, # EMOJI REMOVED
    layout="centered"
)

# Apply custom CSS for a more elegant look, matching the HTML theme
st.markdown("""
    <style>
    /* Gold theme for Streamlit */
    .stApp {
        background-color: #FBF7F0; /* Creamy White */
    }
    .main-header {
        color: #A3885C; /* Primary Gold */
        font-family: 'Playfair Display', serif;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .st-emotion-cache-1c7yb1q { /* Streamlit chat input container */
        border-top: 1px solid #A3885C;
    }
    .st-emotion-cache-4oy39v { /* Chat container background */
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        padding: 10px;
    }
    .stChatMessage.stChatMessage--user { /* User message bubble */
        background-color: #3B3B3B; /* Deep Charcoal */
        color: white;
        border-radius: 10px;
    }
    .stChatMessage.stChatMessage--assistant { /* Assistant message bubble */
        background-color: #EAE3D6; /* Light Cream/Beige */
        color: #1E1E1E;
        border-left: 3px solid #A3885C; /* Gold accent */
        border-radius: 10px;
    }
    
    /* --- ULTIMATE HIDING CSS (Targets multiple potential classes/IDs) --- */
    
    /* 1. HIDE FOOTER (Built with Streamlit) - Targets the common footer tag and test ID */
    footer { visibility: hidden; }
    [data-testid="stFooter"] { visibility: hidden !important; }
    
    /* 2. HIDE FULLSCREEN BUTTON & MENU - Targets the header, top-right menu, and sidebar open button */
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* More aggressive targeting for the menu button and related elements */
    [data-testid="stToolbar"], [data-testid="stSidebarToggleButton"] {
        display: none !important;
    }

    /* Fallback for the old footer class */
    .st-emotion-cache-czk5ad { visibility: hidden; }
    
    </style>
""", unsafe_allow_html=True)


# EMOJI REMOVED
st.markdown('<h1 class="main-header">Golden Spoon AI Assistant</h1>', unsafe_allow_html=True) 
st.caption("Ask about the menu, hours, or general inquiries.")


# Initialize the agent once
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_gemini_agent()

agent_executor = st.session_state.agent_executor

# Initialize chat history in session state
if "messages" not in st.session_state:
    # EMOJI REMOVED
    st.session_state.messages = [
        AIMessage(
            content="Welcome to Golden Spoon Restaurant! I am your AI Assistant. How can I help you today?"
        )
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("Ask about the menu or reservations..."):
    if not agent_executor:
        st.warning("Cannot process request: Agent failed to initialize.")
    else:
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the Agent
        with st.spinner("Thinking..."):
            try:
                # Get the full history for the agent to use
                history_for_agent = st.session_state.messages

                # Invoke the Agent
                response = agent_executor.invoke(
                    {"input": prompt, "chat_history": history_for_agent}
                )

                # Extract Agent Output (FIXED: Handling both failure and success cases)
                if 'output' in response: 
                    full_response = response["output"]
                else:
                    # Fallback if the agent returns an unexpected format
                    print(f"Agent returned unexpected format: {response}")
                    full_response = "Sorry, I ran into an internal error. Please try again."

            except Exception as e:
                # Handle connection errors or other unexpected exceptions
                print(f"Agent Invocation Error: {e}")
                # Assign a safe string value to full_response in case of error
                full_response = "I apologize, a system error occurred while processing your request. Please try refreshing or ask a simpler question."

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(full_response)

            # Add the assistant's response to the chat history
            st.session_state.messages.append(AIMessage(content=full_response))
