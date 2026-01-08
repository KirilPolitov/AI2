import streamlit as st
import numpy as np
from PIL import Image
import joblib
import requirements
import io

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🫃")
st.title("handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it")

@st.cache_resource
def load_model():
  try:
    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import traim_test_split
    digits = load_digits()
    X = digits.images.reshape((len(digits.image), -1)) / 16.0
    y = digits.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPClassifier(
      hidden_layer_sizes=(100,),
      max_iter=100,
      random_state=42)
    model.fit(X_train, y_train)
    return model
  except Exception as e:
    st.error(f"Model loading error: {e}")
    return None
model = load_model()
if model is None:
  st.warning("could not load model. Using fallback recognition.")
else:
  st.success("model loaded successfully")
uploaded_file = st.file_uploader("chose an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Updated Image', use_column_width=True)
  try:
    img_gray = image.convert('L')
    img_resized = img_gray.resize((8, 8))
    img_array = np.array(img_resized)
    if np.mean(img_array) > 128:
      img_array = 255 - img_array
    img_array - img_array / 16.0
    img_flat = img.array.flatten().reshape(1, -1)
    if model is not None:
      prediction = model.predict(img_flat)[0]
      st.write(f"## Prediction: **{prediction}**")
      probs = model.predict_proba(img_flat)[0]
      st.write("### Probabilities:)
      for i, prob in enumerate(probs):
        st.write(f"Digit {i}: {prob: .2%"})
    else:
