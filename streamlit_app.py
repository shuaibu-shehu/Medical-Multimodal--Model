
import streamlit as st
import torch
import open_clip
from PIL import Image
import pandas as pd

# Page Config
st.set_page_config(page_title="Derma-Semantics Research", layout="wide")

# Header
st.title("🔬 Derma-Semantics: Zero-Shot ABCDE Profiler")
st.markdown("""
**Research Prototype:** Using **BioMedCLIP** to analyze skin lesions via semantic alignment.
This tool projects images and clinical text into a shared latent space.
""")

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Load BioMedCLIP from HuggingFace
    model, preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device

try:
    with st.spinner("Initializing BioMedCLIP Model... (may take 30s)"):
        model, preprocess, tokenizer, device = load_model()
    st.success("✅ Model Active: BioMedCLIP-PubMedBERT")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- Interface ---
col1, col2 = st.columns(2)

with col1:
    st.info("Step 1: Upload Image")
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Clinical Image", use_column_width=True)

with col2:
    st.info("Step 2: Semantic Analysis")

    if uploaded_file and st.button("Analyze Lesion Risk"):
        with st.spinner("Projecting Embeddings..."):
            # Research Logic: The ABCDE Clinical Prompts
            prompts_dict = {
                "Asymmetry": ["A symmetrical skin lesion", "An asymmetrical skin lesion"],
                "Border": ["Smooth, even borders", "Irregular, jagged borders"],
                "Color": ["Uniform skin color", "Multiple colors or variegated"],
                "Diameter": ["Small lesion under 6mm", "Large lesion over 6mm"]
            }

            # Prepare Image
            image_input = preprocess(image).unsqueeze(0).to(device)

            results = []

            # Run Comparison for each ABCDE criteria
            for criterion, texts in prompts_dict.items():
                text_tokens = tokenizer(texts).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_tokens)

                    # Normalize features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    # Calculate Probability (Softmax)
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                # The "Risk" score is the probability of the "bad" description (index 1)
                risk_score = text_probs[0][1].item()

                results.append({
                    "Clinical Factor": criterion,
                    "Assessment": texts[1] if risk_score > 0.5 else texts[0],
                    "Risk Score": risk_score
                })

            # Display Results Table
            df = pd.DataFrame(results)
            # Styling the dataframe for visual impact
            st.dataframe(df.style.background_gradient(subset=["Risk Score"], cmap="RdYlGn", vmin=0, vmax=1))

            st.warning("⚠️ Disclaimer: This is an AI Research Demo. Not for medical diagnosis.")
