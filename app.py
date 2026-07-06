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


# Traduccion palabra por palabra: el dataset Fruits-360 tiene mas de 200 clases
# (variedades como "Apple Braeburn", "Tomato Cherry Red", etc.), asi que en vez de
# traducir cada combinacion a mano, se traduce cada palabra conocida y se arma
# el nombre completo en español.
TRADUCCIONES = {
    "apple": "Manzana", "apricot": "Albaricoque", "avocado": "Aguacate",
    "banana": "Banana", "beetroot": "Remolacha", "blueberry": "Arándano",
    "cactus": "Cactus", "fruit": "Fruta", "cantaloupe": "Melón Cantalupo",
    "carambula": "Carambola", "cauliflower": "Coliflor", "cherry": "Cereza",
    "wax": "Cera", "chestnut": "Castaña", "clementine": "Clementina",
    "cocos": "Coco", "corn": "Maíz", "husk": "Cáscara", "cucumber": "Pepino",
    "ripe": "Maduro", "dates": "Dátiles", "eggplant": "Berenjena", "fig": "Higo",
    "ginger": "Jengibre", "root": "Raíz", "granadilla": "Granadilla",
    "grape": "Uva", "grapefruit": "Toronja", "guava": "Guayaba",
    "hazelnut": "Avellana", "huckleberry": "Arándano Silvestre", "kaki": "Caqui",
    "kiwi": "Kiwi", "kohlrabi": "Colinabo", "kumquats": "Kumquat",
    "lemon": "Limón", "meyer": "Meyer", "limes": "Lima", "lychee": "Lichi",
    "mandarine": "Mandarina", "mango": "Mango", "mangostan": "Mangostán",
    "maracuja": "Maracuyá", "melon": "Melón", "piel": "Piel", "de": "de",
    "sapo": "Sapo", "mulberry": "Mora", "nectarine": "Nectarina",
    "flat": "Plana", "nut": "Nuez", "forest": "Silvestre", "pecan": "Pecana",
    "onion": "Cebolla", "peeled": "Pelada", "orange": "Naranja",
    "papaya": "Papaya", "passion": "Pasión", "peach": "Durazno",
    "pear": "Pera", "abate": "Abate", "forelle": "Forelle", "kaiser": "Kaiser",
    "monster": "Monster", "stone": "Piedra", "williams": "Williams",
    "pepino": "Pepino Dulce", "pepper": "Pimiento", "green": "Verde",
    "yellow": "Amarillo", "physalis": "Physalis", "with": "con",
    "pineapple": "Piña", "mini": "Mini", "pitahaya": "Pitahaya",
    "plum": "Ciruela", "pomegranate": "Granada", "pomelo": "Pomelo",
    "sweetie": "Sweetie", "potato": "Papa", "washed": "Lavada",
    "sweet": "Dulce", "quince": "Membrillo", "rambutan": "Rambután",
    "raspberry": "Frambuesa", "redcurrant": "Grosella Roja", "salak": "Salak",
    "strawberry": "Fresa", "wedge": "Gajo", "tamarillo": "Tomate de Árbol",
    "tangelo": "Tangelo", "tomato": "Tomate", "heart": "Corazón",
    "maroon": "Granate", "not": "no", "ripened": "Maduro", "walnut": "Nuez",
    "watermelon": "Sandía", "red": "Rojo", "pink": "Rosa", "white": "Blanco",
    "blue": "Azul", "black": "Negro", "golden": "Dorada", "braeburn": "Braeburn",
    "granny": "Granny", "smith": "Smith", "crimson": "Crimson", "snack": "Snack",
    "rainier": "Rainier", "lady": "Lady", "finger": "Finger",
}


def formatear_nombre(nombre_clase):
    """Traduce el nombre de la clase original del dataset (ej. 'Apple Braeburn')
    a español, palabra por palabra, usando el diccionario TRADUCCIONES. Las
    palabras que no aparecen en el diccionario (por ejemplo nombres propios de
    variedades) se dejan tal cual, solo con formato de título."""
    palabras = nombre_clase.replace("_", " ").strip().split()
    traducidas = [TRADUCCIONES.get(p.lower(), p.title()) for p in palabras]
    return " ".join(traducidas)


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
    st.caption("Arleth Adyani Chevez Bonilla — Cuenta: 20221900251")
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
