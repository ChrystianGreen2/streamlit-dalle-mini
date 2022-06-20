import streamlit as st
import generator
text = st.text_input('Insira o texto', 'Dog riding a bike')
number = int(st.number_input('Insert a number'))

if st.button("submit") and text and number and isinstance(number, int):
    print(f"Creating {number} images of {text}\n")
    images = generator.get_images(text, n_predictions=number, gen_top_k=None, gen_top_p=None, temperature=None)
    if images:
        st.image(images)
        st.write('The current movie title is', text)
