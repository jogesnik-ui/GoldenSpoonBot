import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- (A) Custom CSS for Professional Branding ---
# We use st.markdown to inject CSS, giving the app a distinct, non-generic look.
def apply_custom_css():
    st.markdown(
        """
        <style>
        /* 1. Global Theming for a professional look */
        /* Set the main background to a soft off-white */
        .stApp {
            background-color: #FAFAFA; /* Off-White/Light Gray */
        }
        
        /* 2. Style the Assistant's Chat Bubbles (SpoonBot) */
        /* Target the chat message content div for the assistant */
        [data-testid="stChatMessage"][data-state="visible"]:nth-child(even) [data-testid="stChatMessageContent"] {
            background-color: #F8F4E3; /* Soft light gold for the assistant's reply */
            border-left: 5px solid #A87A4F; /* Golden brown accent line */
            color: #333333; /* Dark gray text for contrast */
            border-radius: 0.75rem;
            padding: 1rem;
        }

        /* 3. Style the User's Chat Bubbles */
        /* Target the chat message content div for the user */
        [data-testid="stChatMessage"][data-state="visible"]:nth-child(odd) [data-testid="stChatMessageContent"] {
            background-color: #E3E3E3; /* Light gray for the user's input */
            color: #1A1A1A; /* Very dark text */
            border-radius: 0.75rem;
            padding: 1rem;
        }

        /* 4. Customizing Header and Chat Input */
        /* Change the main app title color (using the primary color as the accent) */
        .st-emotion-cache-1jm6hrg h1 {
            color: #8B0000; /* Deep Maroon/Red */
        }
        
        /* Change the input box color for a better look */
        div.st-emotion-cache-1c9v92d {
             background-color: #FFFFFF; /* White input box */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- (1) Initial Setup: Load Keys, Initialize LLM/Agent (Runs once) ---

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not google_api_key:
    st.error("FATAL ERROR: GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# Initialize the Language Model (LLM) and Tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2, 
    google_api_key=google_api_key
)

# We keep the search tool available
search_tool = TavilySearch(
    max_results=1, 
    api_key=tavily_api_key
)
tools = [search_tool]


# --- (2) The Key Change: Restaurant-Specific System Prompt ---

RESTAURANT_INFO = (
    "You are 'SpoonBot,' the friendly AI assistant for **The Golden Spoon** restaurant. "
    "Your tone should be warm, professional, and concise. "
    "Your primary goal is to answer customer questions related to the restaurant's services. "
    "NEVER answer questions unrelated to the restaurant. If asked an irrelevant question, politely redirect the user to ask about the restaurant. "
    "Do not use external search (the provided search tool) unless it's strictly necessary for a very specific, real-time detail that is *not* in your known facts (e.g., today's date or a current event). "
    "Your known facts are: "
    "1. **Menu Highlights:** Famous for the 'Golden Steak' ($45) and the 'Chef's Signature Pasta' ($30). We offer vegetarian and gluten-free options. "
    "2. **Hours:** Open Monday to Saturday from 5:00 PM to 10:00 PM. We are closed on Sundays. "
    "3. **Bookings:** Reservations are highly recommended. Guests can book by calling us at (555) 123-GSPD or through our website booking form. "
    "4. **Location:** Downtown at 101 Culinary Lane. "
    "Keep answers short (1-3 sentences) and always maintain a professional, friendly, and helpful demeanor. **Use emojis sparingly and appropriately, like a $\\text{🍽️}$ or $\\text{🥄}$ at the end of a message.**"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RESTAURANT_INFO),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- (3) Streamlit Interface Setup ---

# Apply the CSS first!
apply_custom_css()

st.set_page_config(page_title="The Golden Spoon Bot 🥄", layout="centered")
st.title("The Golden Spoon AI Concierge 🍽️") 
st.caption("Ask me about our menu, hours, or how to book a table!")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an opening welcome message from the assistant
    st.session_state.messages.append({"role": "assistant", "content": "Welcome! I'm SpoonBot, your assistant for The Golden Spoon. How may I help you today? 🥄"})

# Display previous chat messages
for message in st.session_state.messages:
    # IMPORTANT FIX: Use st.markdown inside st.chat_message to prevent weird text formatting
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- (4) Handle User Input and Generate Response ---

if prompt_input := st.chat_input("Ask me a question about The Golden Spoon..."):
    # 1. Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input) # Use st.markdown here too

    # 2. Prepare chat history for the Agent (LangChain format)
    agent_history = []
    for message in st.session_state.messages:
        if message["role"] == "user" and message["content"] != prompt_input:
            agent_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            agent_history.append(AIMessage(content=message["content"]))
    
    # 3. Get the Agent's response
    try:
        with st.chat_message("assistant"):
            with st.spinner("SpoonBot is checking our details..."):
                response = agent_executor.invoke({
                    "input": prompt_input,
                    "chat_history": agent_history
                })
        
        agent_response = response["output"]

    except Exception as e:
        agent_response = f"I apologize, but I encountered an unexpected error: {e}"
        st.error(agent_response)


    # 4. Display the Agent's final response and add to history
    if 'output' in response:
        # We display the final response using st.markdown for consistent formatting
        final_response_placeholder = st.empty() 
        final_response_placeholder.markdown(agent_response)
        st.session_state.messages.append({"role": "assistant", "content": agent_response})