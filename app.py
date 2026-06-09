import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model = tf.saved_model.load("waste_model_savedmodel")
    return model.signatures["serving_default"]

INFER = load_model()

IMG_SIZE = (224, 224)

# ==========================
# TAMPILAN
# ==========================
st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Klasifikasi Sampah")
st.write("Upload gambar sampah untuk mengetahui jenisnya.")

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    if st.button("Analisis"):

        img = image.resize(IMG_SIZE)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0).astype(np.float32)

        start = time.time()

        input_tensor = tf.constant(arr)

        output = INFER(input_tensor)

        output_key = list(output.keys())[0]
        pred = output[output_key].numpy()

        score = float(pred[0][0])

        # Jika hasil terbalik nanti ubah ke:
        # is_organic = score < 0.5

        is_organic = score >= 0.5

        label = "Organic" if is_organic else "Inorganic"

        confidence = round(
            score * 100 if is_organic
            else (1 - score) * 100,
            2
        )

        elapsed = round(
            (time.time() - start) * 1000,
            1
        )

        st.success(f"Hasil: {label}")

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

        st.metric(
            "Waktu Inferensi",
            f"{elapsed} ms"
        )

        if label == "Organic":

            st.info("""
            Sampah organik berasal dari makhluk hidup dan dapat terurai alami.

            • Buat kompos dari sisa makanan  
            • Buang ke tempat sampah organik  
            • Bisa dijadikan pupuk tanaman
            """)

        else:

            st.info("""
            Sampah anorganik sulit terurai dan perlu didaur ulang.

            • Pisahkan plastik, kaca, dan kertas  
            • Bawa ke bank sampah  
            • Kurangi plastik sekali pakai
            """)