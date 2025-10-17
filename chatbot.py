"""
Empathetic Conversational Chatbot - Streamlit App
Complete single-file implementation with all decoding strategies
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tokenizers import Tokenizer
import re
from datetime import datetime
from pathlib import Path
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.classes.*')


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Empathetic AI Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 1.5rem 3rem;
        max-width: 1400px;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #666;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    # .chat-container {
    #     background: white;
    #     border-radius: 20px;
    #     padding: 1.5rem;
    #     box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    #     min-height: 450px;
    #     max-height: 450px;
    #     overflow-y: auto;
    #     margin-bottom: 1rem;
    # }
    
    .chat-message {
        display: flex;
        margin-bottom: 1.2rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        justify-content: flex-end;
    }
    
    .bot-message {
        justify-content: flex-start;
    }
    
    .message-content {
        max-width: 70%;
        padding: 0.9rem 1.3rem;
        border-radius: 18px;
        word-wrap: break-word;
    }
    
    .user-message .message-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 5px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .bot-message .message-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-bottom-left-radius: 5px;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
    }
    
    .message-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin: 0 0.65rem;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .message-time {
        font-size: 0.7rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 0.2rem;
    }
    
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 18px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.4rem;
        background: rgba(255, 255, 255, 0.2);
    }
    
    .method-badge {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.3);
        margin-left: 0.4rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.8rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.4rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 0.9rem;
        border-radius: 12px;
        color: white;
        margin: 0.8rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.65rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .example-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 1rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
        width: 100%;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    .example-btn:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.5);
    }
    
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .loading-dots span {
        animation: blink 1.4s infinite both;
        display: inline-block;
    }
    
    .loading-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .loading-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes blink {
        0%, 80%, 100% { opacity: 0; }
        40% { opacity: 1; }
    }
    
    /* Hide default Streamlit elements */
    div[data-testid="stToolbar"] {
        display: none;
    }
    
    /* Compact input areas */
    .stTextArea textarea {
        font-size: 0.95rem;
    }
    /* --- WELCOME MESSAGE BOX STYLE --- */
    .welcome-box {
    /* Background: Slightly lighter than the main app background for contrast */
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 12px;
    padding: 3rem;
    margin: 2rem 0; /* Ensures it sits neatly within the column */
    text-align: center;
    box-shadow: 0 0 20px #FFD7001A; /* Soft gold glow */
    }

    .welcome-box h3 {
    color: #FFD700 !important; /* Gold header */
    font-weight: 800;
    font-family: 'Montserrat', sans-serif;
    margin-bottom: 0.5rem !important;
    }

    .welcome-box p {
    color: #F0F8FF !important; /* Light text for body */
    opacity: 0.85;
    margin-top: 0.5rem;
    }
    .stTextInput input {
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL ARCHITECTURE CLASSES (Same as before - no changes)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        return torch.matmul(attn_probs, V), attn_probs
    
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output)), attn_probs

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))

class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=2, num_encoder_layers=2, 
                 num_decoder_layers=2, d_ff=1024, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return src_mask, tgt_mask
    
    def encode(self, src, src_mask):
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.output_projection(dec_output)

# ============================================================================
# LOAD MODEL & TOKENIZER (Same as before)
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """Load model and tokenizer from local directory"""
    try:
        current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
        
        tokenizer_paths = [
            current_dir / 'empathetic_tokenizer.json',
            current_dir / 'results' / 'empathetic_tokenizer.json'
        ]
        
        model_paths = [
            current_dir / 'final_model.pt',
            current_dir / 'results' / 'final_model.pt',
            current_dir / 'best_model.pt',
            current_dir / 'results' / 'best_model.pt'
        ]
        
        tokenizer_path = None
        for path in tokenizer_paths:
            if path.exists():
                tokenizer_path = path
                break
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        if not tokenizer_path or not model_path:
            st.error(f"""
            **Files not found!**
            
            Looking for:
            - Tokenizer: {[str(p) for p in tokenizer_paths]}
            - Model: {[str(p) for p in model_paths]}
            """)
            return None, None, {}, "error"
        
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        checkpoint = torch.load(
            str(model_path), 
            map_location='cpu',
            weights_only=False
        )
        
        model = TransformerChatbot(
            vocab_size=checkpoint['vocab_size'],
            **checkpoint['model_config']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, tokenizer, checkpoint.get('test_results', {}), "local"
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, {}, "error"

# ============================================================================
# INFERENCE FUNCTIONS (Same as before)
# ============================================================================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()

def greedy_decode(model, src, max_len=50):
    model.eval()
    with torch.no_grad():
        src_mask, _ = model.generate_mask(src, src)
        enc_output = model.encode(src, src_mask)
        tgt = torch.tensor([[1]])
        
        for _ in range(max_len):
            _, tgt_mask = model.generate_mask(src, tgt)
            dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
            output = model.output_projection(dec_output[:, -1, :])
            next_token = output.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == 2:
                break
        
        return tgt.squeeze(0).tolist()

def greedy_decode_with_penalty(model, src, max_len=50, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        src_mask, _ = model.generate_mask(src, src)
        enc_output = model.encode(src, src_mask)
        tgt = torch.tensor([[1]])
        generated_tokens = []
        
        for _ in range(max_len):
            _, tgt_mask = model.generate_mask(src, tgt)
            dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
            logits = model.output_projection(dec_output[:, -1, :]).squeeze(0)
            
            for token in set(generated_tokens):
                logits[token] = logits[token] / repetition_penalty
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            
            if next_token.item() == 2:
                break
        
        return tgt.squeeze(0).tolist()

def nucleus_sampling_decode(model, src, max_len=50, p=0.9, temperature=1.0):
    model.eval()
    with torch.no_grad():
        src_mask, _ = model.generate_mask(src, src)
        enc_output = model.encode(src, src_mask)
        tgt = torch.tensor([[1]])
        
        for _ in range(max_len):
            _, tgt_mask = model.generate_mask(src, tgt)
            dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
            logits = model.output_projection(dec_output[:, -1, :]).squeeze(0)
            
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == 2:
                break
        
        return tgt.squeeze(0).tolist()

def beam_search_decode(model, src, beam_width=3, max_len=50):
    model.eval()
    with torch.no_grad():
        src_mask, _ = model.generate_mask(src, src)
        enc_output = model.encode(src, src_mask)
        beams = [([1], 0.0)]
        
        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[-1] == 2:
                    candidates.append((seq, score))
                    continue
                
                tgt = torch.tensor([seq])
                _, tgt_mask = model.generate_mask(src, tgt)
                dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
                output = model.output_projection(dec_output[:, -1, :])
                
                log_probs = F.log_softmax(output, dim=-1)
                top_probs, top_indices = log_probs.topk(beam_width)
                
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    candidates.append((seq + [idx.item()], score + prob.item()))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq[-1] == 2 for seq, _ in beams):
                break
        
        return beams[0][0]

def generate_response(model, tokenizer, input_text, method='greedy', **kwargs):
    """Generate response with input length validation"""
    try:
        input_enc = tokenizer.encode(normalize_text(input_text))
        input_ids = [1] + input_enc.ids + [2]
        
        # Check if input is too long
        MAX_LENGTH = 128  # Model's maximum sequence length
        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"Input is too long ({len(input_ids)} tokens). "
                f"Maximum allowed is {MAX_LENGTH} tokens. "
                f"Please shorten your message or situation."
            )
        
        src = torch.tensor([input_ids], dtype=torch.long)
        
        if method == 'greedy':
            output_ids = greedy_decode(model, src)
        elif method == 'greedy_penalty':
            output_ids = greedy_decode_with_penalty(model, src, repetition_penalty=kwargs.get('repetition_penalty', 1.2))
        elif method == 'nucleus':
            output_ids = nucleus_sampling_decode(model, src, p=kwargs.get('p', 0.9), temperature=kwargs.get('temperature', 1.0))
        elif method == 'beam':
            output_ids = beam_search_decode(model, src, beam_width=kwargs.get('beam_width', 3))
        else:
            output_ids = greedy_decode(model, src)
        
        output_tokens = [tokenizer.id_to_token(id) for id in output_ids if id not in [0, 1, 2]]
        output_text = ' '.join(output_tokens)
        output_text = output_text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        
        return output_text
        
    except ValueError as e:
        # User-friendly error message
        return f"❌ Error: {str(e)}"
    except Exception as e:
        # Generic error message
        return f"❌ Sorry, I encountered an error processing your request. Please try with a shorter message. (Error: {type(e).__name__})"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Load model
    with st.spinner("🔄 Loading AI Model..."):
        model, tokenizer, test_results, load_source = load_model_and_tokenizer()
    
    if model is None:
        st.error("❌ Failed to load model")
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💙 Empathetic AI Chatbot</h1>
        <p>Transformer Architecture • Built from Scratch • Advanced Decoding</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0
    if 'current_situation' not in st.session_state:
        st.session_state.current_situation = ""
    if 'current_message' not in st.session_state:
        st.session_state.current_message = ""
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = "happy"
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Emotion selector
        emotion_map = {
            "😊 Happy": "happy",
            "😢 Sad": "sad",
            "😠 Angry": "angry",
            "😨 Afraid": "afraid",
            "😮 Surprised": "surprised",
            "🤢 Disgusted": "disgusted",
            "🎉 Excited": "excited",
            "💭 Sentimental": "sentimental",
            "😰 Anxious": "anxious",
            "😌 Proud": "proud",
            "🙏 Grateful": "grateful",
            "😔 Lonely": "lonely",
            "😄 Content": "content",
            "😒 Jealous": "jealous"
        }
        
        # Find index for current emotion
        default_index = list(emotion_map.values()).index(st.session_state.current_emotion) if st.session_state.current_emotion in emotion_map.values() else 0
        
        selected_emotion_display = st.selectbox(
            "Select Emotion",
            list(emotion_map.keys()),
            index=default_index,
            key="emotion_selector"
        )
        selected_emotion = emotion_map[selected_emotion_display]
        
        st.markdown("---")
        
        # Decoding settings
        st.markdown("### 🎯 Decoding Strategy")
        
        decoding_methods = {
            "🚀 Greedy (Fastest)": "greedy",
            "🎯 Greedy + Penalty (Recommended)": "greedy_penalty",
            "🌟 Nucleus Sampling (Diverse)": "nucleus",
            "💎 Beam Search (Best Quality)": "beam"
        }
        
        selected_method_display = st.selectbox(
            "Method",
            list(decoding_methods.keys()),
            index=1
        )
        selected_method = decoding_methods[selected_method_display]
        
        # Method-specific parameters
        st.markdown("#### Parameters")
        repetition_penalty = 1.2
        p_value = 0.9
        temperature = 1.0
        beam_width = 3
        
        if selected_method == 'greedy_penalty':
            repetition_penalty = st.slider(
                "Repetition Penalty", 
                1.0, 2.0, 1.2, 0.1,
                help="Higher = less repetition"
            )
        elif selected_method == 'nucleus':
            p_value = st.slider(
                "Top-p (Nucleus)", 
                0.5, 1.0, 0.9, 0.05,
                help="Lower = more focused"
            )
            temperature = st.slider(
                "Temperature", 
                0.5, 1.5, 1.0, 0.1,
                help="Higher = more random"
            )
        elif selected_method == 'beam':
            beam_width = st.slider(
                "Beam Width", 
                2, 5, 3, 1,
                help="More beams = better quality"
            )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Info")
        st.markdown("""
        <div class="info-box">
        <b>Architecture:</b><br>
        • 2 Encoder Layers<br>
        • 2 Decoder Layers<br>
        • 2 Attention Heads<br>
        • 256 Embedding Dim<br>
        • ~15M Parameters
        </div>
        """, unsafe_allow_html=True)
        
        # Test metrics
        if test_results:
            st.markdown("### 📈 Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">BLEU</div>
                    <div class="metric-value">{test_results.get('bleu', 0):.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">chrF</div>
                    <div class="metric-value">{test_results.get('chrf', 0):.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ROUGE-L</div>
                    <div class="metric-value">{test_results.get('rouge_l', 0):.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Perplexity</div>
                    <div class="metric-value">{test_results.get('perplexity', 0):.1f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Session")
        st.info(f"""
        **Conversations:** {st.session_state.conversation_count}  
        **Messages:** {len(st.session_state.messages)}
        """)
        
        # Clear button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            st.session_state.current_situation = ""
            st.session_state.current_message = ""
            st.session_state.current_emotion = "happy"
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-box">
                <h3>👋 Welcome! Start a conversation</h3>
                <p>Select an emotion, describe your situation, and send a message below.</p>
                <p style='font-size: 0.9rem; margin-top: 1.5rem; opacity: 0.7;'>
                    💡 Click a Quick Load Scenario on the right to get started!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display messages
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-content">
                            <div class="emotion-badge">{msg.get('emotion', 'unknown')}</div>
                            <div>{msg['content']}</div>
                            <div class="message-time">{msg.get('time', '')}</div>
                        </div>
                        <div class="message-avatar user-avatar">👤</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    method_badge = f"<span class='method-badge'>{msg.get('method', 'N/A')}</span>"
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-avatar bot-avatar">🤖</div>
                        <div class="message-content">
                            <div>{msg['content']}</div>
                            <div class="message-time">{msg.get('time', '')} {method_badge}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input container
        st.markdown("---")
        
        # Situation input
        situation = st.text_area(
            "📝 Describe the situation",
            value=st.session_state.current_situation,
            placeholder="What's happening? Provide some context...",
            height=70,
            key="situation_input_field"
        )
        
        # Message input and send button
        col_msg, col_btn = st.columns([4, 1])
        with col_msg:
            user_message = st.text_input(
                "💬 Your message",
                value=st.session_state.current_message,
                placeholder="Type your message here...",
                key="message_input_field"
            )
        with col_btn:
            send_button = st.button("Send 🚀", type="primary", use_container_width=True)
        
        # Handle send
        if send_button:
            if user_message and situation:
                # Add user message
                current_time = datetime.now().strftime("%H:%M")
                st.session_state.messages.append({
                    'role': 'user',
                    'content': user_message,
                    'emotion': selected_emotion,
                    'situation': situation,
                    'time': current_time
                })
                
                # Generate response
                with st.spinner("🤔 Thinking..."):
                    input_text = f"emotion: {selected_emotion} | situation: {situation} | customer: {user_message} agent:"
                    
                    kwargs = {}
                    if selected_method == 'greedy_penalty':
                        kwargs['repetition_penalty'] = repetition_penalty
                    elif selected_method == 'nucleus':
                        kwargs['p'] = p_value
                        kwargs['temperature'] = temperature
                    elif selected_method == 'beam':
                        kwargs['beam_width'] = beam_width
                    
                    response = generate_response(model, tokenizer, input_text, method=selected_method, **kwargs)
                
                # Check if response is an error
                if response.startswith("❌"):
                    # Show error to user without adding to chat
                    st.error(response)
                    # Remove the user message we just added
                    st.session_state.messages.pop()
                else:
                    # Add bot response
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': response,
                        'time': datetime.now().strftime("%H:%M"),
                        'method': selected_method.upper().replace('_', '+')
                    })
                    
                    st.session_state.conversation_count += 1
                    
                    # Clear inputs after sending
                    st.session_state.current_situation = ""
                    st.session_state.current_message = ""
                    
                    st.rerun()
            else:
                st.warning("⚠️ Please provide both situation and message!")
    
    with col2:
        # Example prompts with auto-fill
        st.markdown("### 💡 Quick Examples")
        st.markdown("*Click to auto-fill*")
        
        examples = [
            {
                "emotion": "sad",
                "situation": "i've been feeling down about my job search rejection",
                "message": "i just feel like i'm not good enough for anything",
                "emoji": "😢",
                "title": "Job Rejection"
            },
            {
                "emotion": "excited",
                "situation": "i'm going on vacation to the mountains next week with friends",
                "message": "i've been looking forward to this for months!",
                "emoji": "🎉",
                "title": "Vacation Plans"
            },
            {
                "emotion": "afraid",
                "situation": "the weather forecast predicts a huge storm hitting my town tonight",
                "message": "i'm really worried about the damage it might cause",
                "emoji": "😨",
                "title": "Storm Warning"
            },
            {
                "emotion": "angry",
                "situation": "my roommate broke my favorite coffee mug and didn't even apologize",
                "message": "i'm so mad, it was a limited edition!",
                "emoji": "😠",
                "title": "Broken Mug"
            },
            {
                "emotion": "grateful",
                "situation": "my best friend stayed up all night to help me prepare for my exam",
                "message": "i don't know what i'd do without their support",
                "emoji": "🙏",
                "title": "Supportive Friend"
            },
            {
                "emotion": "lonely",
                "situation": "i just moved to a new city for work and don't know anyone here",
                "message": "everyone seems to have their own friend groups already",
                "emoji": "😔",
                "title": "New City"
            },
            {
                "emotion": "proud",
                "situation": "i finally completed my first marathon after months of training",
                "message": "i can't believe i actually did it!",
                "emoji": "😌",
                "title": "Marathon Achievement"
            }
        ]
        
        for idx, ex in enumerate(examples):
            if st.button(
                f"{ex['emoji']} {ex['title']}",
                key=f"ex_{idx}",
                use_container_width=True
            ):
                # Update session state with example data
                st.session_state.current_situation = ex['situation']
                st.session_state.current_message = ex['message']
                st.session_state.current_emotion = ex['emotion']
                st.rerun()
        
        st.markdown("---")
        
        # Method comparison
        with st.expander("🔬 Decoding Methods"):
            st.markdown("""
            **🚀 Greedy**
            - Fastest (~50ms)
            - Most likely token
            
            **🎯 Greedy + Penalty**
            - No repetition
            - ⭐ Recommended!
            
            **🌟 Nucleus**
            - Diverse outputs
            - Creative responses
            
            **💎 Beam Search**
            - Best quality
            - Slower (~150ms)
            """)
        
        # Tips
        with st.expander("💡 Usage Tips"):
            st.markdown("""
            **For best results:**
            - Click examples to auto-fill
            - Match emotion to situation
            - Be specific and detailed
            - Try different methods
            
            **⚠️ Limitations:**
            - Maximum input length: ~100 words
            - Keep messages concise and clear
            - Model trained on conversational text
            
            **Parameters:**
            - Repetition: 1.2-1.5
            - Top-p: 0.85-0.95
            - Temperature: 0.8-1.2
            - Beam Width: 3-5
            """)
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            ### Empathetic Chatbot
            
            **Built from scratch:**
            - Transformer architecture
            - No pretrained models
            - 4 decoding strategies
            
            **Technologies:**
            - PyTorch
            - Streamlit
            - HF Tokenizers
            
            **Dataset:**
            Empathetic Dialogues (Facebook AI)
            """)

# Footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; padding: 0.8rem; color: #666;'>
#     <p style='margin: 0; font-size: 0.9rem;'><b>Empathetic AI Chatbot</b> | PyTorch & Streamlit</p>
#     <p style='margin: 0.3rem 0 0 0; font-size: 0.8rem;'>Transformer • No Pretrained Models • Advanced Decoding</p>
# </div>
# """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()