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
def forward(X, params):
    A = X
    L = len(params) // 2

    for l in range(1, L):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        A = relu(Z)

    # Output layer
    ZL = A @ params[f"W{L}"] + params[f"b{L}"]
    AL = sigmoid(ZL)

    return AL

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

        y_hat = forward(X, params)
        prob = float(y_hat[0][0])

        label = CLASS_NAMES[int(prob > 0.5)]

        st.success(f"✅ Prediction: **{label}**")
        st.write(f"Confidence: **{prob:.4f}**")

