from openai import OpenAI
import streamlit as st
import time
import json
import os
from datetime import datetime

# Page config and styling
st.set_page_config(
    page_title="NVIDIA AI Chat",
    page_icon="🤖",
    layout="wide",
)

# Apply custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stTextInput>div>div>input {
        background-color: #1E1E1E;
    }
    .clear-button {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to save conversation to file
def save_conversation(history, filename=None):
    if not os.path.exists("e:/nlp/nvidia/conversations"):
        os.makedirs("e:/nlp/nvidia/conversations")
    
    if filename is None:
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    filepath = f"e:/nlp/nvidia/conversations/{filename}"
    with open(filepath, "w") as f:
        json.dump(history, f, indent=4)
    
    return filepath

# Function to load conversation from file
def load_conversation(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

# Function to display chat messages
def display_messages(messages):
    for message in messages:
        if message["role"] == "system":
            continue  # Skip system messages in display
        
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = "new_chat"
if 'saved_conversations' not in st.session_state:
    # Scan for existing conversations
    if not os.path.exists("e:/nlp/nvidia/conversations"):
        os.makedirs("e:/nlp/nvidia/conversations")
    files = os.listdir("e:/nlp/nvidia/conversations")
    st.session_state.saved_conversations = {f.replace(".json", ""): f for f in files if f.endswith(".json")}

# Sidebar - Configuration
with st.sidebar:
    st.title("Chat Settings")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["nvidia/llama-3.1-nemotron-ultra-253b-v1", "nvidia/mixtral-8x7b-instruct-v0.1"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt:", 
        value="You are a helpful, detailed, and knowledgeable assistant.",
        height=100
    )
    
    st.divider()
    
    # Conversation management
    st.subheader("Conversation Management")
    
    # New chat button
    if st.button("New Chat", key="new_chat"):
        st.session_state.chat_history = []
        st.session_state.current_conversation = "new_chat"
        st.experimental_rerun()
    
    # Save current conversation
    if st.button("Save Current Chat", key="save_chat"):
        if st.session_state.chat_history:
            save_name = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filepath = save_conversation(st.session_state.chat_history, f"{save_name}.json")
            st.session_state.saved_conversations[save_name] = os.path.basename(filepath)
            st.session_state.current_conversation = save_name
            st.success(f"Conversation saved as {save_name}")
    
    # Load saved conversations
    st.subheader("Saved Conversations")
    if st.session_state.saved_conversations:
        selected_conversation = st.selectbox(
            "Select a conversation to load",
            options=list(st.session_state.saved_conversations.keys()),
            index=None
        )
        
        if selected_conversation and st.button("Load Conversation"):
            filepath = f"e:/nlp/nvidia/conversations/{st.session_state.saved_conversations[selected_conversation]}"
            st.session_state.chat_history = load_conversation(filepath)
            st.session_state.current_conversation = selected_conversation
            st.experimental_rerun()
    else:
        st.info("No saved conversations found.")

# Main interface
st.title("NVIDIA AI Chat Assistant")
st.caption("Powered by NVIDIA's Nemotron Ultra & Mixtral Models")

# Initialize API client
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

# Display chat history
if st.session_state.chat_history:
    display_messages([msg for msg in st.session_state.chat_history if msg["role"] != "system"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat
    st.chat_message("user").write(user_input)
    
    # Add to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Set up messages including system prompt and history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history (up to last 10 messages to avoid context length issues)
    for msg in st.session_state.chat_history[-10:]:
        if msg["role"] != "system":
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Create assistant message placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the response
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    full_response += content_chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
            # Auto-save if using a named conversation
            if st.session_state.current_conversation != "new_chat":
                save_conversation(
                    st.session_state.chat_history, 
                    f"{st.session_state.current_conversation}.json"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Clear chat button (bottom of main panel)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Clear Chat", key="clear_chat_main"):
        st.session_state.chat_history = []
        st.session_state.current_conversation = "new_chat"
        st.experimental_rerun()

