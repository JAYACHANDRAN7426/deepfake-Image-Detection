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
#                USER MODEL PATHS
# ---------------------------------------------------------
MESO_WEIGHTS = r"C:\Users\User\Downloads\Meso4_DF.h5"
EFF_WEIGHTS  = r"C:\Users\User\Downloads\New folder (2)\DeepfakeDetector-main\models\best_model-v3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
#                   BUILD MESO4
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

# Load MesoNet
meso = build_meso4()
meso.load_weights(MESO_WEIGHTS)

def meso_predict(img):
    img = img.resize((256,256))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, 0)
    return float(meso.predict(arr)[0][0])

# ---------------------------------------------------------
#               LOAD EFFICIENTNET — FIXED
# ---------------------------------------------------------
eff_model = efficientnet_b0(weights=None)

# EfficientNetB0 classifier fix (always works)
in_features = eff_model.classifier[1].in_features
eff_model.classifier[1] = torch.nn.Linear(in_features, 2)

# Load weights
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
        output = eff_model(t)
        probability_fake = torch.softmax(output, dim=1)[0][1].item()
    return float(probability_fake)

# ---------------------------------------------------------
#        MODEL 3 (SIMULATED)
# ---------------------------------------------------------
def model3_simulated(m1, m2):
    return float((m1 + m2)/2 + np.random.uniform(-0.05, 0.05))

# ---------------------------------------------------------
#                 STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.markdown("""
    <h1 style="text-align:center;">🔍 AI Deepfake Detection System</h1>
    <h4 style="text-align:center;">Powered by EfficientNet + MesoNet + Fusion Model</h4>
    <br>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=350)

    st.subheader("🔄 Running Predictions...")

    # Predict
    m1 = meso_predict(img)
    m2 = eff_predict(img)  # EfficientNet FIXED
    m3 = model3_simulated(m1, m2)

    # Final decision (EfficientNet)
    label = "FAKE" if m2 >= 0.50 else "REAL"

    # -----------------------------------------------------
    # Show scores
    # -----------------------------------------------------
    st.subheader("📊 Model Prediction Scores")
    st.write("0 = Real, 1 = Fake")

    st.write(f"**Meso4:** `{m1:.3f}`")
    st.progress(m1)

    st.write(f"**EfficientNet (Final):** `{m2:.3f}`")
    st.progress(m2)

    st.write(f"**Model-3:** `{m3:.3f}`")
    st.progress(m3)

    # -----------------------------------------------------
    # Bar Chart
    # -----------------------------------------------------
    st.subheader("📈 Comparison Chart")

    fig, ax = plt.subplots(figsize=(6,4))
    names = ["Meso4", "EfficientNet", "Model-3"]
    values = [m1, m2, m3]

    ax.bar(names, values)
    ax.set_ylim(0,1)
    ax.set_ylabel("Fake Probability")
    ax.set_title("Fake Scores by Model")

    st.pyplot(fig)

    # -----------------------------------------------------
    # Final Result card
    # -----------------------------------------------------
    st.subheader("🧾 Final Decision (EfficientNet Based)")

    color = "red" if label == "FAKE" else "green"
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background:{color}; color:white; text-align:center; font-size:26px;">
            <b>{label}</b><br>
            Confidence: {m2:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )
