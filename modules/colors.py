# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:26:36 2025

@author: quent
"""
import streamlit as st
import numpy as np
import plotly.express as px

# Choose-colors is a function that enable the user to choose the color for each category of the footprint
# The idea is to give this choice once (in the home page) and then to keep the same for each graph
# A similar function will exist to choose the color of each solution unless we want to create a link


def choose_colors(categories):
    st.subheader("Choose a color for each main category")

    if "category_colors" not in st.session_state:
        st.session_state.category_colors = {}

    for cat in categories:
        # Generate only if no color has ever been assigned
        if cat not in st.session_state.category_colors:
            random_color = "#" + ''.join(np.random.choice(list("0123456789ABCDEF"), 6))
            st.session_state.category_colors[cat] = random_color

        # Always use current value from session for color picker default
        color = st.color_picker(
            f"Color for {cat}",
            value=st.session_state.category_colors[cat],
            key=f"color_picker_{cat}"
        )

        st.session_state.category_colors[cat] = color


# show_pie_chart_by_category creates a simple pie chart including all the main categories of emissions with their colors

def show_pie_chart_by_category(dataframe, title="Emission breakdown by main category"):
    if "Category" not in dataframe.columns or "Emissions" not in dataframe.columns:
        st.warning("The columns 'Category' or 'Emissions' are missing.")
        return

    # Aggregate emissions by main category
    emissions_by_cat = dataframe.groupby("Category")["Emissions"].sum().reset_index()
    
    # Get saved color mapping
    color_mapping = st.session_state.get("category_colors", {})
    
    
    # Plotly pie chart
    fig = px.pie(
        emissions_by_cat,
        names="Category",
        values="Emissions",
        title=title,
        hole=0.3,
        color="Category",  # this triggers use of color_discrete_map
        color_discrete_map=color_mapping
    )
    st.plotly_chart(fig, use_container_width=True)