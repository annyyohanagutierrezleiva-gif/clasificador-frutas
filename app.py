import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "modelo_frutas_mobilenet/frutas_mobilenet.h5"
CLASSES_PATH = "modelo_frutas_mobilenet/class_name.json"

st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍎", layout="wide")


@st.cache_resource
def load_model():
    """Carga el modelo y las clases una sola vez (gracias al cache de Streamlit),
    para no tener que recargarlos cada vez que el usuario sube una imagen."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            class_name = json.load(f)
        return model, class_name
    except FileNotFoundError:
        st.error(
            "No se encontraron los archivos del modelo. Verifica que la carpeta "
            f"'modelo_frutas_mobilenet' con '{MODEL_PATH.split('/')[-1]}' y "
            f"'{CLASSES_PATH.split('/')[-1]}' esté junto a app.py."
        )
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")
        st.stop()


def formatear_nombre(nombre_clase):
    """Convierte el nombre de la carpeta original del dataset (ej. 'Apple Braeburn')
    en un texto un poco más prolijo para mostrar en la interfaz. Con más de 200
    clases no es práctico traducir cada variedad al español, así que se muestra
    el nombre original con formato de título."""
    return nombre_clase.replace("_", " ").strip().title()


def predict_image(model, class_name, img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1]
    return [(class_name[i], preds[i]) for i in top_idx]


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("🍇 Clasificador de Frutas")
    st.caption("Anny Yohana Gutierrez — Cuenta: 20211930078")
    st.divider()
    st.write("**Modelo:** MobileNetV2 (Transfer Learning)")
    st.write("**Dataset:** Fruits-360 (Kaggle)")
    st.divider()
    st.info(
        "Este modelo fue entrenado con imágenes de estudio del dataset "
        "Fruits-360 (fondo uniforme, fruta centrada y sola). Con fotos "
        "reales tomadas en un entorno natural la predicción puede ser "
        "menos precisa."
    )

model, class_name = load_model()

# ---------- Encabezado ----------
st.title("Clasificador de Frutas con MobileNetV2")
st.write(
    "Sube una imagen de una fruta o verdura (el dataset Fruits-360 incluye ambas) "
    "y el modelo predecirá su categoría."
)

st.divider()

# ---------- Carga de imagen y resultados ----------
col_img, col_res = st.columns(2)

with col_img:
    st.subheader("📷 Imagen")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
        except Exception:
            st.error("No se pudo abrir la imagen. Probá con otro archivo JPG o PNG.")
            st.stop()

        st.image(img, caption="Imagen cargada", use_container_width=True)

with col_res:
    st.subheader("🔍 Resultado")

    if uploaded_file is None:
        st.write("Aquí aparecerá la predicción una vez que subas una imagen.")
    else:
        with st.spinner("Analizando imagen..."):
            results = predict_image(model, class_name, img)

        top_class, top_prob = results[0]

        st.metric("Predicción", formatear_nombre(top_class), f"{top_prob*100:.2f}% de confianza")

        st.write("**Top 5 probabilidades**")
        for raw, prob in results[:5]:
            st.write(f"{formatear_nombre(raw)} — {prob*100:.2f}%")
            st.progress(float(prob))
