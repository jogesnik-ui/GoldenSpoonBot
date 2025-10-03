import streamlit as st
import os
from dotenv import load_dotenv
import random
from datetime import datetime # Import datetime for current date and formatting

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder 
from langchain_core.tools import tool # Import the tool decorator

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

# --- NEW: Tool Definitions for Booking Simulation ---

@tool
def get_current_date() -> str:
    """Returns the current date and time in Australian DD/MM/YYYY format for accurate temporal reference (e.g., 'today', 'tomorrow')."""
    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    time_str = now.strftime("%I:%M %p")
    return f"Current date is {date_str} and current time is {time_str}."

@tool
def check_and_book_reservation(date: str, time: str, party_size: int) -> str:
    """
    Checks the restaurant's internal booking system for table availability.
    Requires a specific date (e.g., '2025-10-15'), a time (e.g., '7:00 PM'), and the number of people (e.g., 4).
    The tool returns availability status, NOT an automatic confirmation.
    """
    try:
        # Attempt to parse date, assuming LLM provides YYYY-MM-DD or similar
        # If the LLM provides '2025-10-15', this handles it.
        date_obj = datetime.strptime(date.split('T')[0], "%Y-%m-%d") if 'T' in date else datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d/%m/%Y")
    except ValueError:
        # If date parsing fails, use the raw string provided by the user/LLM
        formatted_date = date
        
    # Party Size Restriction Check
    if party_size > 6:
        return f"We regret to inform you that we can only process online availability checks for parties of 6 or fewer. Please call (555) 123-4567 to inquire about availability for a party of {party_size}."
    
    # Simulate availability (randomly fail about 40% of the time)
    is_available = random.random() >= 0.4
    
    if is_available:
        # Do NOT automatically confirm. Provide availability details and prompt for confirmation.
        available_slots = ["6:30 PM", "7:45 PM", "8:00 PM"]
        
        # Check if the requested time is one of the available slots
        if time in available_slots:
            # If the requested time is available, state it clearly
            return f"Great news! We have availability for a party of {party_size} on {formatted_date} at {time}. To confirm and finalize this reservation, please reply with 'Yes, book the table.'"
        else:
            # If the requested time is NOT available, offer alternatives
            return f"We do not have a table available at {time} on {formatted_date} for {party_size} people, but we have tables available at: {', '.join(available_slots)}. Would you like to book one of these times instead?"
    else:
        # Simulate being fully booked
        return f"We apologize, but we are fully booked for a party of {party_size} on {formatted_date}. Please try a different time or date, or call us directly at (555) 123-4567 for late cancellations."

# --- System Prompt Definition (Updated for Tool Use) ---
SYSTEM_PROMPT = """
You are the Golden Spoon Restaurant AI. Your job is to help customers with reservations, menu inquiries, and general questions about the Golden Spoon Restaurant.

**Your Identity and Role:**
* You are a professional, polite, and friendly assistant.
* Your responses should be concise and elegant, matching the restaurant's upscale branding.
* **SCOPE RESTRICTION:** You MUST only answer questions related to the Golden Spoon Restaurant. If a user asks a question unrelated to the restaurant, politely state: "I apologize, but my purpose is to assist you with inquiries regarding the Golden Spoon Restaurant only."
* You have access to a real-time search tool (Tavily Search) for up-to-date information. Use it only when necessary.
* **TEMPORAL CONTEXT:** When a user asks a relative time question (e.g., 'today', 'tomorrow', 'next week'), you MUST use the `get_current_date` tool first to establish the current context.
* **RESERVATIONS:** * When the user asks to check availability or make a reservation, you MUST use the `check_and_book_reservation` tool first.
    * The tool returns availability status, NOT a final confirmation. Always relay the tool's message to the user and follow its instructions (e.g., prompt the user to confirm).
* **RESTAURANT KNOWLEDGE:**
    * **Name:** Golden Spoon Restaurant
    * **Cuisine:** Classic French techniques blended with local, seasonal ingredients.
    * **Signature Dishes:** Seared Scallops, Signature Filet Mignon, Handmade Lobster Ravioli, Decadent Lava Cake.
    * **Atmosphere:** Luxurious and welcoming.
    * **Hours:** Dinner: Tues - Sat, 5:00 PM - 10:00 PM | Brunch: Sun, 10:00 AM - 2:00 PM.
* Do not invent information. If the answer is not in your knowledge, the tool's output, or search results, politely decline.
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

    # 2. Define Tools (ADD THE NEW BOOKING TOOL AND DATE TOOL)
    tavily_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=3)
    tools = [tavily_tool, check_and_book_reservation, get_current_date]

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
    page_icon=None,
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
    
    /* --- ULTIMATE HIDING CSS --- */
    footer { display: none !important; }
    [data-testid="stFooter"] { display: none !important; }
    #MainMenu { display: none !important; } 
    header { display: none !important; } 
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stSidebarToggleButton"] { display: none !important; }
    .st-emotion-cache-czk5ad { display: none !important; }
    
    </style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">Golden Spoon AI Assistant</h1>', unsafe_allow_html=True) 
st.caption("Ask about the menu, hours, or general inquiries.")


# Initialize the agent once
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_gemini_agent()

agent_executor = st.session_state.agent_executor

# Initialize chat history in session state
if "messages" not in st.session_state:
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
