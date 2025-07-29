# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:38:30 2025

@author: quent
"""
import streamlit as st
from streamlit_tree_select import tree_select
from modules.tree import build_tree

def create_growth(years):
    """
    Create a new growth-based or budget-based projection and store it in session state.
    
    This function displays a form where the user can define a projection by either
    specifying a fixed growth percentage or a budget evolution over time. The form
    supports optional intermediate budget points and stores the result in session state.
    
    Parameters:
    - years (List[int]): List of target years, including the start and end years for the projection.
    
    Effects:
    - Updates st.session_state["growth_inputs"] with a new projection object containing:
        - 'name': The label of the projection.
        - 'mode': Either "Growth %" or "Budget Projection".
        - 'growth': Growth percentage (if applicable).
        - 'budget': Dictionary of budgets per year (if applicable).
        - 'categories': Placeholder for future tree-based category assignment.
    - Displays success confirmation in the Streamlit interface.
    """
    
    st.subheader("Create a new growth or budget projection")

    if "growth_inputs" not in st.session_state:
        st.session_state.growth_inputs = []

    start_year = years[0]
    end_year = years[-1]
    intermediate_years = [y for y in years if y != start_year and y != end_year]

    with st.form("create_growth_form"):
        name = st.text_input("Name of the projection (e.g. 'General org', 'Events')")
        mode = st.radio("Type of projection", ["Growth %", "Budget Projection"])

        growth = None
        budget = {}

        if mode == "Growth %":
            growth = st.number_input("Growth percentage (%)", min_value=0.0, format="%.2f")
        else:
            budget_start = st.number_input(f"Budget in {start_year} (€)", min_value=0.0, format="%.2f")
            budget_end = st.number_input(f"Budget in {end_year} (€)", min_value=0.0, format="%.2f")

            selected_intermediates = st.multiselect(
                "Add optional intermediate years",
                intermediate_years
            )

            for y in selected_intermediates:
                val = st.number_input(
                    f"Budget in {y} (€)",
                    min_value=0.0,
                    format="%.2f",
                    key=f"budget_{name}_{y}"
                )
                budget[y] = val

            budget[str(start_year)] = budget_start
            budget[str(end_year)] = budget_end

        submitted = st.form_submit_button("Add projection")

        if submitted and name:
            new_proj = {
                "name": name,
                "mode": mode,
                "growth": growth,
                "budget": budget,
                "categories": {}  # placeholder for tree selection
            }

            st.session_state.growth_inputs.append(new_proj)
            st.success(f"✅ Projection '{name}' added.")


def assign_growth(data):
    """
    Allow the user to assign categories to existing growth or budget projections.
    
    This function displays each existing projection in a form, using a two-column layout
    for better visual balance. Users can review projection details and assign categories
    through an interactive tree selector.
    
    Parameters:
    - data (pd.DataFrame): Emissions data used to construct the hierarchical tree structure.
    
    Effects:
    - For each projection in st.session_state["growth_inputs"], displays a form with:
        - Projection name and type (Growth % or Budget).
        - Existing growth or budget values.
        - A tree selector for assigning categories.
    - Updates the 'categories' field of each projection based on user selections.
    - Displays confirmation messages when changes are saved.
    """
    st.subheader("Assign growth or budget projections to categories")

    if "growth_inputs" not in st.session_state or not st.session_state.growth_inputs:
        st.info("No projections available.")
        return

    tree = build_tree(data)
    col1, col2 = st.columns(2)

    for i, g in enumerate(st.session_state.growth_inputs):
        # Alternate between columns
        col = col1 if i % 2 == 0 else col2

        with col.form(f"assign_growth_{i}"):
            st.markdown(f"### 🛠️ {g['name']} ({g['mode']})")

            if g["mode"] == "Growth %":
                st.write(f"Growth: **{g['growth']}%**")
            else:
                st.write("Budget projection:")
                for year, amount in sorted(g["budget"].items(), key=lambda x: int(x[0])):
                    st.markdown(f"- **{year}**: {amount:,.0f} €")

            checked = g["categories"].get("checked", []) if isinstance(g["categories"], dict) else []
            selection = tree_select(tree, checked=checked, key=f"growth_tree_{i}")
            g["categories"] = selection

            submitted = st.form_submit_button("Save configuration")
            if submitted:
                st.success("✅ Categories updated.")



def apply_projections_to_base(projection_df, years):
    """
    Apply growth or budget projections to the projection DataFrame based on user-defined inputs.

    For each growth or budget projection stored in session state, this function updates
    the Value_YEAR columns of rows whose full hierarchical path matches the selected categories.
    It supports two modes:
    - Growth %: Applies exponential growth from the start year.
    - Budget Projection: Applies year-by-year budget scaling, with interpolation or extrapolation
      if some years are missing.

    Parameters:
    - projection_df (pd.DataFrame): DataFrame containing one row per item, with Value_YEAR columns.
    - years (List[int]): List of target years (e.g., [2025, 2026, ..., 2035]).

    Returns:
    - pd.DataFrame: Updated projection DataFrame with adjusted Value_YEAR values.
    """
    if "growth_inputs" not in st.session_state:
        return projection_df

    start_year = years[0]

    for g in st.session_state.growth_inputs:
        selected_paths = set(g.get("categories", {}).get("checked", []))

        for idx, row in projection_df.iterrows():
            full_path = row["Full path"]

            if full_path in selected_paths:

                # === Mode 1: Growth %
                if g["mode"] == "Growth %":
                    base_val = row[f"Value_{start_year}"]
                    for y in years[1:]:
                        factor = (1 + g["growth"] / 100) ** (y - start_year)
                        projection_df.at[idx, f"Value_{y}"] = base_val * factor

                # === Mode 2: Budget Projection
                elif g["mode"] == "Budget Projection":
                    budget = {str(int(float(k))): v for k, v in g.get("budget", {}).items()}
                    known_years = sorted(int(y) for y in budget if budget[y] > 0)

                    if not known_years:
                        continue  # no usable budget

                    for y in years:
                        y_str = str(y)
                        base_value = row[f"Value_{start_year}"]

                        if y_str in budget:
                            ratio = budget[y_str] / budget[str(start_year)]
                            projection_df.at[idx, f"Value_{y}"] = base_value * ratio
                        else:
                            # Interpolate or extrapolate
                            before_candidates = [k for k in known_years if k < y]
                            after_candidates = [k for k in known_years if k > y]

                            if before_candidates and after_candidates:
                                before = max(before_candidates)
                                after = min(after_candidates)
                                v_before = budget[str(before)]
                                v_after = budget[str(after)]

                                ratio = (y - before) / (after - before)
                                interpolated_budget = v_before + ratio * (v_after - v_before)
                                projected_value = base_value * (interpolated_budget / budget[str(start_year)])
                                projection_df.at[idx, f"Value_{y}"] = projected_value

                            elif before_candidates:
                                before = max(before_candidates)
                                projected_value = base_value * (budget[str(before)] / budget[str(start_year)])
                                projection_df.at[idx, f"Value_{y}"] = projected_value

                            elif after_candidates:
                                after = min(after_candidates)
                                projected_value = base_value * (budget[str(after)] / budget[str(start_year)])
                                projection_df.at[idx, f"Value_{y}"] = projected_value

                            else:
                                # fallback: no info → keep start value
                                projection_df.at[idx, f"Value_{y}"] = base_value

    return projection_df


def check_projection_coverage(projected_df):
    """
    Verify that each row in the projection table is covered by exactly one growth or budget projection.

    This function checks whether every item in the projected DataFrame is associated with
    a single projection based on the assigned category tree. It identifies two types of issues:
    - Rows without any assigned projection.
    - Rows matched by multiple projections.

    Parameters:
    - projected_df (pd.DataFrame): DataFrame containing the projection data, including a 'Full path' column.

    Effects:
    - Displays Streamlit warnings for rows with missing or overlapping projections.
    - Shows a maximum of 30 individual warnings for readability.
    - Displays a success message if all rows are correctly covered.
    """
    warnings = []

    if "growth_inputs" not in st.session_state:
        return

    for idx, row in projected_df.iterrows():
        full_path = row["Full path"]
        name = row["Name"]

        matching_growths = 0

        for g in st.session_state.growth_inputs:
            selected = set(g.get("categories", {}).get("checked", []))
            if full_path in selected:
                matching_growths += 1

        if matching_growths == 0:
            warnings.append(f"⚠️ No projection applied to: **{full_path} > {name}**")
        elif matching_growths > 1:
            warnings.append(f"⚠️ Multiple projections applied to: **{full_path} > {name}**")

    if warnings:
        st.warning("Some rows have missing or conflicting projections:")
        for w in warnings[:30]:  # Limit display
            st.markdown(f"- {w}")
        if len(warnings) > 30:
            st.markdown(f"...and {len(warnings) - 30} more.")
    else:
        st.success("✅ All rows are correctly covered by exactly one growth/budget projection.")
