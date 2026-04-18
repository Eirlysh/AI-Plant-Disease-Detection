import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google import genai
import os

# =========================
# PAGE CONFIG (FULL WIDTH UI)
# =========================
st.set_page_config(
    page_title="AI Plant Disease Detection",
    layout="wide",
    page_icon="🌿"
)

# =========================
# API KEY
# =========================
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Chưa có GEMINI_API_KEY")
    st.stop()

client = genai.Client(api_key=API_KEY)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CLASS NAMES
# =========================
class_names = [
    'Apple_scab','Apple_black_rot','Apple_rust','Apple_healthy',
    'Background','Blueberry_healthy','Cherry_powdery_mildew','Cherry_healthy',
    'Corn_gray_leaf_spot','Corn_rust','Corn_blight','Corn_healthy',
    'Grape_black_rot','Grape_esca','Grape_leaf_blight','Grape_healthy',
    'Orange_greening','Peach_bacterial_spot','Peach_healthy',
    'Pepper_bacterial_spot','Pepper_healthy',
    'Potato_early_blight','Potato_late_blight','Potato_healthy',
    'Raspberry_healthy','Soybean_healthy','Squash_mildew',
    'Strawberry_scorch','Strawberry_healthy',
    'Tomato_bacterial_spot','Tomato_early_blight','Tomato_late_blight',
    'Tomato_mold','Tomato_septoria','Tomato_spider_mite',
    'Tomato_target_spot','Tomato_yellow_leaf','Tomato_mosaic',
    'Tomato_healthy'
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 39)

    checkpoint = torch.load("final_model.pth", map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# GEMINI CALL
# =========================
def call_gemini(prompt):
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest"
    ]

    for m in models_to_try:
        try:
            res = client.models.generate_content(
                model=m,
                contents=prompt
            )
            return res.text
        except:
            continue

    return "❌ Không gọi được Gemini"

# =========================
# PREDICT FUNCTION
# =========================
def predict(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        top5_prob, top5_idx = torch.topk(prob, 5)

    results = []
    for i in range(5):
        results.append((
            class_names[top5_idx[0][i].item()],
            top5_prob[0][i].item()
        ))

    return results

# =========================
# UI HEADER
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🌿 AI Bệnh Cây Trồng</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SESSION STATE (AUTO REFRESH)
# =========================
if "results" not in st.session_state:
    st.session_state.results = None
if "answer" not in st.session_state:
    st.session_state.answer = None
# =========================
# UPLOAD IMAGE
# =========================
uploaded = st.file_uploader(
    "📤 Upload ảnh lá cây",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    # dùng bytes để detect file thay đổi chính xác hơn name
    file_bytes = uploaded.getvalue()

    # kiểm tra ảnh mới
    if (
        "last_file" not in st.session_state
        or st.session_state.last_file != file_bytes
    ):
        st.session_state.last_file = file_bytes

        # reset kết quả cũ khi upload ảnh mới
        st.session_state.results = None
        st.session_state.answer = None

    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="📷 Ảnh đã upload", use_column_width=True)

    with col2:

        # chỉ chạy predict nếu chưa có kết quả
        if st.session_state.results is None:
            st.info("🤖 Đang phân tích ảnh...")
            results = predict(image)
            st.session_state.results = results
        else:
            results = st.session_state.results

        # =========================
        # DISPLAY RESULTS
        # =========================
        st.subheader("📊 Kết quả dự đoán")

        for i, (name, score) in enumerate(results):
            if i == 0:
                st.success(f"🥇 {name} ({score*100:.2f}%)")
            else:
                st.write(f"{i+1}. {name} ({score*100:.2f}%)")

        if results[0][1] < 0.5:
            st.warning("⚠️ Độ tin cậy thấp - nên chụp ảnh rõ hơn")

        # =========================
        # GEMINI PROMPT
        # =========================
        diseases_list = ", ".join(
            [f"{n} ({p*100:.1f}%)" for n, p in results]
        )

        prompt = f"""
You are an agricultural expert.

IMPORTANT:
- Disease names MUST stay in English
- Explanation in Vietnamese
- Treatment in Vietnamese
- Prevention in Vietnamese

Detected:
{diseases_list}

Focus on highest probability disease.
"""

        # chỉ gọi Gemini khi chưa có answer
        if st.session_state.answer is None:
            with st.spinner("🤖 AI đang tư vấn..."):
                answer = call_gemini(prompt)
                st.session_state.answer = answer
        else:
            answer = st.session_state.answer

        st.subheader("🌱 Tư vấn AI")
        st.success(answer)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit + PyTorch + Gemini</center>",
    unsafe_allow_html=True
)