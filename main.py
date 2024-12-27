import streamlit as st
import langchain_helper

st.title('Restaurant name generator')

cuisine=st.sidebar.selectbox("Pick a cuisine",("Indian","Italian","Mexican","American","Arabic"))

if cuisine:
    response =langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'])
    menu_items=response['menu_items'].split(',')
    st.write("**Menu Items**")
    for item in menu_items:
        st.write("-",item)


