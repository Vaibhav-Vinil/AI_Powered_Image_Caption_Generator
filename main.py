#pip install streamlit


import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
import torch

HF_API_KEY = "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"  # Replace with actual API key

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(output[0], skip_special_tokens=True)

def generate_stylized_caption(caption, style):
    url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"


    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    style_prompts = {
        "humorous": f"Rewrite this caption in a humorous and sarcastic way, keeping it under 20 words: '{caption}'",
        "poetic": f"Turn this caption into a short poetic phrase, keeping it under 20 words: '{caption}'",
        "descriptive": f"Make this caption more vivid and detailed, keeping it under 20 words: '{caption}'",
        "mysterious": f"Make this caption eerie and suspenseful in under 20 words: '{caption}'",
        "minimalistic": f"Condense this caption to be short and impactful, keeping it under 10 words: '{caption}'"
    }
    
    if style not in style_prompts:
        return "❌ Error: Unsupported style."
    
    payload = {"inputs": style_prompts[style]}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        #return response_json[0]["generated_text"].strip() if isinstance(response_json, list) else "❌ Unexpected response format."
        generated_text = response_json[0]["generated_text"].strip()
        return generated_text.split(":")[-1].strip()  # Extract only the generated part

    except requests.exceptions.RequestException as e:
        return f"❌ API request failed - {e}"

st.title("🖼️ AI-Powered Image Captioning")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
style = st.selectbox("Choose a caption style", ["humorous", "poetic", "descriptive", "mysterious", "minimalistic"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    processor, model = load_model()
    base_caption = generate_image_caption(image, processor, model)
    st.write("### Generated Caption:", base_caption)
    
    if st.button("Generate Stylized Caption"):
        stylized_caption = generate_stylized_caption(base_caption, style)
        st.write("### Stylized Caption:", stylized_caption)
