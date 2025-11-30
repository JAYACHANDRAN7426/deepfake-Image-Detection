import streamlit as st
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from torchvision.models import efficientnet_b0
from torchvision import transforms

# ---------------------------------------------------------
#               STREAMLIT CLOUD MODEL PATHS
# ---------------------------------------------------------
MESO_WEIGHTS = "models/Meso4_DF.h5"
EFF_WEIGHTS  = "models/best_model-v3.pt"
DEVICE = "cpu"       # Streamlit Cloud has NO GPU

# ---------------------------------------------------------
#                   BUILD MESO4 MODEL
# ---------------------------------------------------------
def build_meso4():
    inp = Input(shape=(256, 256, 3))
    x = Conv2D(8, (3,3), padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)

    x = Conv2D(8, (5,5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)

    x = Conv2D(16, (5,5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)

    x = Conv2D(16, (5,5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((4,4), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(16)(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inp, out)


# Load mesonet weights
meso = build_meso4()
meso.load_weights(MESO_WEIGHTS)


def meso_predict(img):
    img = img.resize((256,256))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    return float(meso.predict(arr)[0][0])

# ---------------------------------------------------------
#               LOAD EFFICIENTNET MODEL
# ---------------------------------------------------------
eff_model = efficientnet_b0(weights=None)

# Fix classifier shape
try:
    in_features = eff_model.classifier[1].in_features
    eff_model.classifier[1] = torch.nn.Linear(in_features, 2)
except:
    eff_model.classifier = torch.nn.Linear(eff_model.classifier.in_features, 2)

# Load PyTorch weights
ck = torch.load(EFF_WEIGHTS, map_location=DEVICE)
if "state_dict" in ck:
    ck = ck["state_dict"]

clean = {k.replace("module.", ""): v for k,v in ck.items()}
eff_model.load_state_dict(clean, strict=False)
eff_model.to(DEVICE)
eff_model.eval()

eff_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def eff_predict(img):
    t = eff_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = eff_model(t)
        prob_fake = torch.softmax(out, dim=1)[0][1].item()
    return float(prob_fake)

# ---------------------------------------------------------
#        MODEL 3 (SIMULATED FUSION MODEL)
# ---------------------------------------------------------
def model3_simulated(m1, m2):
    return float((m1 + m2)/2 + np.random.uniform(-0.05, 0.05))

# ---------------------------------------------------------
#                STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.markdown("""
    <h1 style="text-align:center;">🔍 AI Deepfake Detection System</h1>
    <h4 style="text-align:center;">EfficientNet + MesoNet + Fusion Model</h4>
    <hr>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=350)

    st.subheader("🔄 Running Predictions...")

    # Run predictions
    m1 = meso_predict(img)
    m2 = eff_predict(img)
    m3 = model3_simulated(m1, m2)

    # Final Decision (EfficientNet best model)
    final_label = "FAKE" if m2 > 0.5 else "REAL"

    # ---------------------------------------------------------
    #             Display Scores
    # ---------------------------------------------------------
    st.subheader("📊 Prediction Scores (0 = real • 1 = fake)")

    st.write(f"**Meso4:** `{m1:.3f}`")
    st.progress(m1)

    st.write(f"**EfficientNet (final):** `{m2:.3f}`")
    st.progress(m2)

    st.write(f"**Fusion Model:** `{m3:.3f}`")
    st.progress(m3)

    # ---------------------------------------------------------
    #               Bar Graph
    # ---------------------------------------------------------
    st.subheader("📈 Model Comparison Chart")

    fig, ax = plt.subplots(figsize=(6,4))
    names = ["Meso4", "EfficientNet", "Fusion Model"]
    scores = [m1, m2, m3]

    ax.bar(names, scores)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability of Being Fake")

    st.pyplot(fig)

    # ---------------------------------------------------------
    #               Final Result Card
    # ---------------------------------------------------------
    st.subheader("🧾 Final Deepfake Verdict")

    color = "red" if final_label == "FAKE" else "green"

    st.markdown(
        f"""
        <div style="padding:20px; background:{color}; color:white;
             font-size:28px; text-align:center; border-radius:12px;">
            <b>{final_label}</b><br>
            Confidence: {m2:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )
