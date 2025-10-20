# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:26:36 2025

@author: quent
"""

import streamlit as st
import numpy as np
import plotly.express as px
import re
import hashlib

import pandas as pd



def _slug_key(text: str) -> str:
    """
    Build a safe, short, and stable key fragment from an arbitrary string.

    - Lowercases and keeps alphanumerics and dashes only
    - Appends a short hash to avoid collisions between similar names
    """
    base = re.sub(r"[^a-zA-Z0-9\-]+", "-", str(text).strip().lower()).strip("-")
    h = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:6]
    return f"{base}-{h}" if base else h




def choose_colors(categories, key_prefix: str = "colors", allow_reset: bool = True, allow_reorder: bool = True):
    """
    Display a color picker for each category, allow reordering (Move up/down),
    and store colors + order in Streamlit session state.

    Parameters
    ----------
    categories : List[str]
        Categories for which to choose colors.
    key_prefix : str, optional
        Prefix for widget keys.
    allow_reset : bool, optional
        Whether to show the "Reset all colors" button.
    allow_reorder : bool, optional
        Whether to show Move up/down buttons to change order.
    """

    st.subheader("🎨 Choose colors and order of categories")

    # Initialize color and order state
    if "category_colors" not in st.session_state:
        st.session_state.category_colors = {}
    if "category_order" not in st.session_state:
        st.session_state.category_order = list(categories)

    # Keep order in sync with new categories (add missing, remove obsolete)
    order = st.session_state.category_order
    for cat in categories:
        if cat not in order:
            order.append(cat)
    order = [cat for cat in order if cat in categories]
    st.session_state.category_order = order

    # Palette defaults
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Plotly + px.colors.qualitative.D3
    def _random_hex():
        return "#" + "".join(np.random.choice(list("0123456789ABCDEF"), 6))

    # Reset colors
    if allow_reset and st.button("↺ Reset all colors", key=f"{key_prefix}_reset_colors"):
        new_map = {}
        for idx, cat in enumerate(order):
            new_map[cat] = palette[idx % len(palette)]
        st.session_state.category_colors = new_map
        st.success("Colors reset to default palette.")

    # Reset order
    if allow_reorder and st.button("↻ Reset order (alphabetical)", key=f"{key_prefix}_reset_order"):
        st.session_state.category_order = sorted(categories)
        st.rerun()

    # Display each category row
    for i, cat in enumerate(st.session_state.category_order):
        # Ensure color exists
        if cat not in st.session_state.category_colors:
            st.session_state.category_colors[cat] = palette[i % len(palette)]

        cols = st.columns([3, 0.6, 0.6])
        with cols[0]:
            color = st.color_picker(
                f"{cat}",
                value=st.session_state.category_colors[cat],
                key=f"{key_prefix}_picker_{_slug_key(cat)}"
            )
            st.session_state.category_colors[cat] = color

        if allow_reorder:
            with cols[1]:
                if st.button("⬆️", key=f"{key_prefix}_up_{i}") and i > 0:
                    order[i - 1], order[i] = order[i], order[i - 1]
                    st.session_state.category_order = order
                    st.rerun()
            with cols[2]:
                if st.button("⬇️", key=f"{key_prefix}_down_{i}") and i < len(order) - 1:
                    order[i + 1], order[i] = order[i], order[i + 1]
                    st.session_state.category_order = order
                    st.rerun()

    st.markdown(
        "<p style='color:grey;font-size:0.9em;'>"
        "💡 The chosen order will define how categories are displayed in charts."
        "</p>",
        unsafe_allow_html=True
    )




def show_pie_chart_by_category(
    dataframe,
    title: str = "Emission breakdown by main category",
    key_prefix: str = "pie"
):
    """
    Display a donut pie chart with category order and custom colors preserved.

    Uses:
    - st.session_state['category_order'] for consistent display order
    - st.session_state['category_colors'] for color mapping
    """
    required = {"Category", "Emissions"}
    missing = required - set(dataframe.columns)
    if missing:
        st.warning(f"The following required columns are missing: {', '.join(sorted(missing))}")
        return

    df = dataframe.copy()
    df["Emissions"] = pd.to_numeric(df["Emissions"], errors="coerce").fillna(0.0)
    emissions_by_cat = df.groupby("Category", dropna=False, as_index=False)["Emissions"].sum()

    # Retrieve saved order and colors
    order = st.session_state.get("category_order", list(emissions_by_cat["Category"]))
    colors = st.session_state.get("category_colors", {})
    order = [cat for cat in order if cat in emissions_by_cat["Category"].values]

    # Plotly pie (order respected)
    fig = px.pie(
        emissions_by_cat,
        names="Category",
        values="Emissions",
        title=title,
        hole=0.3,
        color="Category",
        color_discrete_map=colors,
        category_orders={"Category": order}
    )

    st.plotly_chart(fig, use_container_width=True)

