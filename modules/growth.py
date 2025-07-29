# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:38:30 2025

@author: quent
"""
import streamlit as st
import pandas as pd
from streamlit_tree_select import tree_select
from modules.tree import build_tree, get_label_path

#create_projection_base creates the dataframe with all the year from 2025 to 2035 and the EF and value data that will be impacted by the growths, the solutions and the structural effects





def create_growth(years):
    """
    Create a new growth or budget-based projection.
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
    Allow user to assign categories to existing growth or budget projections.
    """
    st.subheader("Assign growth or budget projections to categories")

    if "growth_inputs" not in st.session_state or not st.session_state.growth_inputs:
        st.info("No projections available.")
        return

    tree = build_tree(data)

    for i, g in enumerate(st.session_state.growth_inputs):
        with st.form(f"assign_growth_{i}"):
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
    Apply growth or budget projections from session state to the projection DataFrame.

    Parameters:
    - projection_df (DataFrame): DataFrame with Value_YEAR columns.
    - years (List[int]): List of projection years, e.g. [2025, 2026, ..., 2035]
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
    Check whether each row in the projection table is covered by exactly one growth or budget projection.
    Show warning messages if some rows are missing projections or have conflicting ones.
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




