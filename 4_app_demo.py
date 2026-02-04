import streamlit as st
import time

st.title("Naval Bot Demo")

st.write("¡Bienvenido a tu primer RAG interactivo!")

# Creamos una barra de carga falsa para probar la interfaz
barra = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    barra.progress(i + 1)

st.success("¡El sistema funciona correctamente!")

# Un botón interactivo
if st.button("Púlsame"):
    st.balloons() # ¡Efectos especiales!