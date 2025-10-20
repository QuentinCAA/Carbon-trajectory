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

    import numpy as np
    import plotly.express as px
    import matplotlib.colors as mcolors

    st.subheader("🎨 Choose colors and order of categories")

    # =========================================================
    # --- INITIALIZATION ---
    # =========================================================
    if "category_colors" not in st.session_state:
        st.session_state.category_colors = {}
    if "category_order" not in st.session_state:
        st.session_state.category_order = list(categories)

    # Keep order synchronized with available categories
    order = st.session_state.category_order
    for cat in categories:
        if cat not in order:
            order.append(cat)
    order = [cat for cat in order if cat in categories]
    st.session_state.category_order = order

    # Default palette (Plotly qualitative)
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Plotly + px.colors.qualitative.D3

    # Helper: generate random hex if needed
    def _random_hex():
        return "#" + "".join(np.random.choice(list("0123456789ABCDEF"), 6))

    # Helper: safely convert any color (rgb(...) → #RRGGBB)
    def normalize_color(color_str: str) -> str:
        if isinstance(color_str, str):
            color_str = color_str.strip()
            if color_str.startswith("rgb"):
                try:
                    nums = [int(x) for x in color_str.strip("rgb() ").split(",")]
                    color_str = mcolors.to_hex([n / 255 for n in nums])
                except Exception:
                    color_str = "#888888"
            elif not color_str.startswith("#"):
                # In case of unexpected input, fallback
                color_str = "#888888"
        else:
            color_str = "#888888"
        return color_str

    # =========================================================
    # --- RESET BUTTONS ---
    # =========================================================
    if allow_reset and st.button("↺ Reset all colors", key=f"{key_prefix}_reset_colors"):
        new_map = {}
        for idx, cat in enumerate(order):
            new_map[cat] = palette[idx % len(palette)]
        st.session_state.category_colors = new_map
        st.success("Colors reset to default palette.")

    if allow_reorder and st.button("↻ Reset order (alphabetical)", key=f"{key_prefix}_reset_order"):
        st.session_state.category_order = sorted(categories)
        st.rerun()

    # =========================================================
    # --- DISPLAY EACH CATEGORY ---
    # =========================================================
    for i, cat in enumerate(st.session_state.category_order):
        # Ensure a valid color exists
        if cat not in st.session_state.category_colors:
            st.session_state.category_colors[cat] = palette[i % len(palette)]

        # Normalize color format (avoid rgb(...) crash)
        current_color = normalize_color(st.session_state.category_colors[cat])
        st.session_state.category_colors[cat] = current_color

        cols = st.columns([3, 0.6, 0.6])
        with cols[0]:
            color = st.color_picker(
                f"{cat}",
                value=current_color,
                key=f"{key_prefix}_picker_{cat.replace(' ', '_')}"
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

    # =========================================================
    # --- FOOTER INFO ---
    # =========================================================
    st.markdown(
        "<p style='color:grey;font-size:0.9em;'>"
        "💡 The chosen order defines how categories are displayed in charts."
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


def show_total_emissions(dataframe, title="🌍 Total CO₂ emissions", unit="tCO₂e"):
    """
    Display the total CO₂ emissions from a dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain an 'Emissions' column with numeric values.
    title : str, optional
        The title displayed above the total value.
    unit : str, optional
        The unit to display next to the total (default = 'tCO₂e').

    Effects
    -------
    Displays a styled card showing the total emissions sum in Streamlit.
    """
    if "Emissions" not in dataframe.columns:
        st.warning("The dataframe must contain an 'Emissions' column.")
        return

    # Convert to numeric safely
    df = dataframe.copy()
    df["Emissions"] = pd.to_numeric(df["Emissions"], errors="coerce").fillna(0.0)
    total_emissions = df["Emissions"].sum()

    # Formatting for readability
    formatted_total = f"{total_emissions:,.0f}".replace(",", " ")  # thin spaces for thousands

    # Display in a styled block
    st.markdown(
        f"""
        <div style="
            background-color:#f6f6f6;
            border:1px solid #ddd;
            border-radius:12px;
            padding:1rem;
            margin-top:0.5rem;
            text-align:center;">
            <h3 style="margin-bottom:0.3rem;">{title}</h3>
            <p style="font-size:1.8rem; font-weight:bold; margin:0;">
                {formatted_total} {unit}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
