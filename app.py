import streamlit  as st
import numpy as np
import cv2
import pickle

# ======================
# CONFIG
# ======================
IMG_SIZE = 32
CLASS_NAMES = ["Class 0", "Class 1"]

# ======================
# LOAD CUSTOM MODEL
# ======================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        params = pickle.load(f)
    return params

params = load_model()

# ======================
# ACTIVATION
# ======================
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ======================
# FORWARD (CUSTOM MLP)
# ======================
def forward(X, para):
    w1, b1, w2, b2 = para["w1"], para["b1"], para["w2"], para["b2"]
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1 , w2) +  b2
    a2 = sigmoid(z2)
    y_hat = a2
    cache = {
        "z1": z1 ,
        "a1": a1 ,
        "a2": a2 ,
        "z2": z2 ,
        "y_hat": y_hat

    }
    return cache

# ======================
# PREPROCESS IMAGE
# ======================
def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.reshape(1, -1)  # FLATTEN
    return image

# ======================
# STREAMLIT UI
# ======================
st.title("📷 Camera Image Prediction App")

st.write("Nhấn **Take Photo** → **Predict**")

camera_image = st.camera_input("Take a photo")

if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Captured Image", use_container_width=True)

    if st.button("🔮 Predict"):
        X = preprocess_image(image)

        cache = forward(X, params)
        prob = float(cache["y_hat"][0][0])

        label = CLASS_NAMES[int(prob > 0.5)]

        st.success(f"✅ Prediction anh thuan: **{label}**")
        st.write(f"Confidence: **{prob:.4f}**")

