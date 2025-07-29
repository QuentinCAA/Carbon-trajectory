# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:26:36 2025

@author: quent
"""
import streamlit as st
import numpy as np
import plotly.express as px

def choose_colors(categories):
    """
    Display a color picker for each main category and store selections in session state.
    
    This function allows users to assign a custom color to each category using Streamlit's
    color picker widget. If a color has not yet been set for a category, a random color
    is generated and used as the default.
    
    Parameters:
    - categories (List[str]): List of main category names for which colors must be selected.
    
    Effects:
    - Updates st.session_state["category_colors"] with the selected color for each category.
    """

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



def show_pie_chart_by_category(dataframe, title="Emission breakdown by main category"):
    """
    Display a pie chart of emissions aggregated by main category using Plotly.
    
    This function checks for the required columns, aggregates emissions per category,
    and displays a donut chart with custom colors (if defined in session state).
    
    Parameters:
    - dataframe (pd.DataFrame): Input data containing at least 'Category' and 'Emissions' columns.
    - title (str, optional): Title of the pie chart. Defaults to "Emission breakdown by main category".
    
    Effects:
    - Displays a pie chart using Streamlit.
    - Uses st.session_state["category_colors"] to apply custom colors per category if available.
    - Shows a warning if required columns are missing.
    """
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