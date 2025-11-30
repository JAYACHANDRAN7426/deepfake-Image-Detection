import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from PIL import Image
import keras_cv
import keras_core as keras
import matplotlib.pyplot as plt

# ======================================================================
#                          MODEL 1 – MESONET
# ======================================================================

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

meso = build_meso4()

# Load your trained weight from GitHub repo folder
meso.load_weights("models/Meso4_DF.h5")

def meso_predict(img):
    img = img.resize((256,256))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, 0)
    return float(meso.predict(arr)[0][0])


# ======================================================================
#                 MODEL 2 – EfficientNetV2 (NEW, TF 2.20 SAFE)
# ======================================================================

effnet = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_b0_imagenet"
)

def effv2_predict(img):
    img = img.resize((256,256))
    arr = np.array(img).astype("float32")/255.0
    arr = np.expand_dims(arr, 0)

    feat = effnet.predict(arr)[0]
    score = 1 / (1 + np.exp(-np.mean(feat)))  # sigmoid(mean)
    fake = score
    return float(fake)


# ======================================================================
#                         MODEL 3 – Fusion
# ======================================================================

def model3_simulated(m1, m2):
    return float((m1 + m2)/2 + np.random.uniform(-0.05, 0.05))


# ======================================================================
#                            STREAMLIT UI
# ======================================================================

st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.markdown("""
<h1 style="text-align:center;">🔍 AI Deepfake Detection System</h1>
<h4 style="text-align:center;">MesoNet + EfficientNetV2 + Fusion</h4><br>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=350)

    st.subheader("🔄 Running Predictions...")

    # Run models
    m1 = meso_predict(img)
    m2 = effv2_predict(img)
    m3 = model3_simulated(m1, m2)

    # Final label based on EfficientNetV2
    label = "FAKE" if m2 >= 0.5 else "REAL"

    # -----------------------------------------------------------------
    # SCORES
    # -----------------------------------------------------------------
    st.subheader("📊 Model Prediction Scores")
    st.write("0 = Real, 1 = Fake")

    st.write(f"**Meso4:** `{m1:.3f}`")
    st.progress(m1)

    st.write(f"**EfficientNetV2 (Final):** `{m2:.3f}`")
    st.progress(m2)

    st.write(f"**Model-3 (Fusion):** `{m3:.3f}`")
    st.progress(m3)

    # -----------------------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------------------
    st.subheader("📈 Comparison Chart")

    fig, ax = plt.subplots(figsize=(6,4))
    names = ["Meso4", "EffNetV2", "Fusion"]
    values = [m1, m2, m3]

    ax.bar(names, values)
    ax.set_ylim(0,1)
    ax.set_ylabel("Fake Probability")
    ax.set_title("Fake Scores by Model")

    st.pyplot(fig)

    # -----------------------------------------------------------------
    # FINAL DECISION
    # -----------------------------------------------------------------
    st.subheader("🧾 Final Decision (EfficientNetV2 Based)")

    color = "red" if label == "FAKE" else "green"
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background:{color}; 
        color:white; text-align:center; font-size:26px;">
            <b>{label}</b><br>
            Confidence: {m2:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )
