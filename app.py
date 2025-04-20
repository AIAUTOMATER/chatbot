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
    
    # Model selection (updated to only show Chat LLMs, Image, Video Gen, and Code Gen models from reputable providers)
    model_categories = {
        "Large Language Chat Models": [
            # NVIDIA, Meta, Google, Microsoft, Qwen, Deepseek, AI21Labs, etc.
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "nvidia/llama-3.3-nemotron-super-49b-v1",
            "nvidia/llama-3.1-nemotron-nano-8b-v1",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "nvidia/llama-3.1-nemotron-51b-instruct",
            "nvidia/llama-3.1-405b-instruct",
            "nvidia/llama-3.1-8b-instruct",
            "nvidia/llama-3.2-3b-instruct",
            "nvidia/llama-3.2-11b-vision-instruct",
            "nvidia/llama-3.2-1b-instruct",
            "nvidia/llama-4-maverick-17b-128e-instruct",    # Meta
            "nvidia/llama-4-scout-17b-16e-instruct",        # Meta
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
            "meta/llama3-70b-instruct",
            "meta/llama3-8b-instruct",
            "ai21labs/jamba-1.5-large-instruct",
            "ai21labs/jamba-1.5-mini-instruct",
            "google/gemma-3-27b-it",
            "google/gemma-3-1b-it",
            "google/gemma-2-27b-it",
            "google/gemma-2-9b-it",
            "google/gemma-7b",
            "google/gemma-2b",
            "microsoft/phi-4-mini-instruct",
            "microsoft/phi-4-multimodal-instruct",
            "microsoft/phi-3.5-mini-instruct",
            "microsoft/phi-3.5-moe-instruct",
            "microsoft/phi-3.5-vision-instruct",
            "microsoft/phi-3-vision-128k-instruct",
            "microsoft/phi-3-mini-4k-instruct",
            "microsoft/phi-3-mini-128k-instruct",
            "microsoft/phi-3-small-8k-instruct",
            "microsoft/phi-3-small-128k-instruct",
            "microsoft/phi-3-medium-4k-instruct",
            "microsoft/phi-3-medium-128k-instruct",
            "microsoft/phi-3.5-mini-instruct",
            "deepseek-ai/deepseek-r1",               # Chat/coding model
            "qwen/qwq-32b",
            "qwen/qwen2-7b-instruct",
            "qwen/qwen2.5-7b-instruct",
            "upstage/solar-10.7b-instruct",
            "databricks/dbrx-instruct",
            "abacusai/dracarys-llama-3.1-70b-instruct",
            "zyphra/zamba2-7b-instruct",
            "thudm/chatglm3-6b",
            "mediatek/breeze-7b-instruct",
            "aisingapore/sea-lion-7b-instruct"
        ],
        "Code Generation & Reasoning Models": [
            # Pure code gen, completion, or chat & code (no retrieval only)
            "nvidia/usdcode",    # USD + Python SOTA LLM
            "deepseek-ai/deepseek-r1-distill-qwen-32b",
            "deepseek-ai/deepseek-r1-distill-qwen-14b",
            "deepseek-ai/deepseek-r1-distill-qwen-7b",
            "deepseek-ai/deepseek-r1-distill-llama-8b",
            "mistralai/mamba-codestral-7b-v0.1",
            "mistralai/mistral-small-24b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "mistralai/mixtral-8x22b-instruct-v0.1",
            "bigcode/starcoder2-7b",
            "bigcode/starcoder2-15b",
            "ibm/granite-34b-code-instruct",
            "ibm/granite-8b-code-instruct",
            "google/codegemma-1.1-7b",
            "google/codegemma-7b",
            "qwen/qwen2.5-coder-32b-instruct",
            "qwen/qwen2.5-coder-7b-instruct",
            "tiiuae/falcon3-7b-instruct",      # SoTA instruct, code & reasoning
            "01-ai/yi-large",
            "nvidia/llama3-chatqa-1.5-70b",
            "nvidia/llama3-chatqa-1.5-8b",
            "meta/llama-3.2-3b-instruct",
            "meta/llama-3.2-1b-instruct",
        ],
        "Text-to-Image Generation": [
            "black-forest-labs/FLUX.1-dev",
            "nvidia/consistory",  # Consistent characters in imgs
            "nvidia/maisi",       # Medical 3D CT latent diffusion
            "stabilityai/stable-diffusion-3-medium",
            "stabilityai/stable-diffusion-xl",
            "stabilityai/sdxl-turbo",
            "briaai/BRIA-2.3",
            "GettyImages/edify-image",
            "Shutterstock/edify-3d", # 3D asset generation, text2img cap
        ],
        "Text-to-Video Generation": [
            "stabilityai/stable-video-diffusion",  # SVD: text/image to video sequences
            "nvidia/cosmos-predict1-7b",          # Vision/modeling video sim from text/image
            "nvidia/cosmos-predict1-5b"
        ]
    }
    
    # Model selector UI
    model_category = st.selectbox(
        "Model Category",
        options=list(model_categories.keys()),
        index=0
    )
    
    model = st.selectbox(
        "Select Model",
        options=model_categories[model_category],
        index=0
    )
    
    # Add model info tooltip (authentic descriptions for in-scope models)
    with st.expander("About Selected Model"):
        model_info = {
            "nvidia/llama-3.1-nemotron-ultra-253b-v1": "NVIDIA's 253B parameter Llama-3.1-based model, excelling at scientific/math reasoning, code, high accuracy instruction following.",
            "nvidia/llama-3.3-nemotron-super-49b-v1": "High efficiency NVIDIA Llama-3.3 variant, SOTA for reasoning, tool use, chat, instructions.",
            "nvidia/llama-3.1-nemotron-nano-8b-v1": "Efficient, agentic and accurate NVIDIA Llama 3.1-based SLM for PC/edge; supports chat, code, reasoning.",
            "black-forest-labs/FLUX.1-dev": "FLUX.1 is a state-of-the-art, enterprise-ready image generation model (diffusion based).",
            "nvidia/consistory": "NVIDIA's model for consistent character creation across multiple images, ideal for storyboards and creative design.",
            "stabilityai/stable-diffusion-3-medium": "Stable Diffusion 3 - Medium: advanced, high-quality text-to-image diffusion model from Stability AI.",
            "stabilityai/stable-video-diffusion": "Stable Video Diffusion (SVD): generates short video sequences from text or single image prompts; Stability AI.",
            "deepseek-ai/deepseek-r1": "Efficient LLM from DeepSeek (open), strong at coding, math, reasoning, chat.",
            "mistralai/mamba-codestral-7b-v0.1": "7B Codestral (Mamba architecture): SOTA model for code generation in multiple languages.",
            "mistralai/mixtral-8x22b-instruct-v0.1": "Mixtral 8x22B (MOE): Leading Mixture-of-Experts LLM for reasoning, code, and advanced chat.",
            "bigcode/starcoder2-7b": "StarCoder2-7B: Next-gen open code LLM from BigCode for advanced code completion, synthesis, and explanations.",
            "bigcode/starcoder2-15b": "StarCoder2-15B: Large, high-quality open model for code generation, understanding, and multi-language programming support.",
            "ibm/granite-34b-code-instruct": "IBM Granite-34B-code-instruct: Generative code model for enterprise-grade code synthesis, completion, and conversion.",
            "ibm/granite-8b-code-instruct": "IBM's efficient 8B code model for code generation, explanation, and multi-turn conversational coding.",
            "google/codegemma-1.1-7b": "Google CodeGemma-1.1-7B: Powerful code generation and reasoning model for Python and more.",
            "google/codegemma-7b": "Google's Gemma-7B model specialized for coding, completions, and reasoning with code.",
            "mistralai/mistral-small-24b-instruct": "24B parameter Mistral model tuned for fast, high-accuracy code generation, instruction following, and reasoning.",
            "qwen/qwq-32b": "Qwen QWQ-32B: Multilingual, reasoning-friendly code and chat model from Alibaba Qwen.",
            "qwen/qwen2.5-coder-32b-instruct": "Qwen2.5-Coder-32B-Instruct: Fast, powerful multi-language code generation model.",
            "qwen/qwen2.5-coder-7b-instruct": "Qwen2.5-7B Coder: Mid-size, high-performance multilingual code model.",
            "deepseek-ai/deepseek-r1-distill-qwen-32b": "DeepSeekR1-distill-Qwen-32B: Enhanced distilled model for code and reasoning (Qwen backbone).",
            "deepseek-ai/deepseek-r1-distill-llama-8b": "Distilled Llama-8B by DeepSeek, for efficient open-coding and reasoning.",
            "tiiuae/falcon3-7b-instruct": "Falcon3-7B-Instruct: SoTA model for coding and functionally rigorous LLM outputs.",
            "nvidia/usdcode": "NVIDIA USDCode: State-of-the-art LLM for answering USD queries and generating high-quality USD-Python code.",
            "stabilityai/stable-diffusion-xl": "Stable Diffusion XL: Industry-leading, photorealistic image synthesis from text prompts.",
            "briaai/BRIA-2.3": "BRIA-2.3: Enterprise-grade text-to-image diffusion model, trained for creative high-fidelity image generation.",
            "GettyImages/edify-image": "Getty Images Edify: 4K image generation model trained on Getty's commercial, rights-compliant library.",
            "Shutterstock/edify-3d": "Shutterstock Edify-3D: Generative 3D asset and image creation model for design and visualization.",
            "nvidia/maisi": "MAISI: NVIDIA's CT-volumetric text-to-image diffusion model for medical imaging.",
            "nvidia/cosmos-predict1-7b": "NVIDIA Cosmos-Predict1-7B: Generates physics-aware video/world states from text or image prompts, relevant in physical and synthetic AI domains.",
            "nvidia/cosmos-predict1-5b": "NVIDIA Cosmos-Predict1-5B: Efficient physics-aware video future frame prediction from image/video prompts.",
            "stabilityai/sdxl-turbo": "SDXL-Turbo: Near-instant, single-pass, photorealistic text-to-image synthesis with impressive detail.",
            "nvidia/llama3-chatqa-1.5-70b": "NVIDIA Llama3-ChatQA-1.5-70B: Context-wise chatbot model, advanced text and code reasoning (non-commercial use).",
            "nvidia/llama3-chatqa-1.5-8b": "NVIDIA Llama3-ChatQA-1.5-8B: Lightweight Chat LLM for high-quality response generation (non-commercial use).",
            "meta/llama-3.3-70b-instruct": "Meta Llama-3.3-70B: Leading open LLM for chat, reasoning, general knowledge, and code/function calling.",
            "meta/llama3-70b-instruct": "Meta's SOTA 70B Llama3 for advanced chat, code, data analysis, and complex reasoning.",
            "ai21labs/jamba-1.5-large-instruct": "AI21 Labs Jamba-1.5 Large: Fluent, MOE-based LLM for multilingual chat, analysis, and creative tasks.",
        }
        
        if model in model_info:
            st.info(model_info[model])
        else:
            st.info("A leading open or NVIDIA-supported generative model for chat, image, video, or code. See official documentation or Model Zoo for full details.")
        
        st.caption("Note: Model availability and commercial usage may vary based on your NVIDIA API/cloud provider access. Some models may require licenses or specific usage rights.")
    
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
        st.rerun()
    
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
            st.rerun()
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
        st.rerun()

