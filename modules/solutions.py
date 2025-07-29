# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 08:21:30 2025

@author: quent
"""

import streamlit as st
from modules.tree import build_tree
from streamlit_tree_select import tree_select
import pandas as pd
import numpy as np


def init_solutions():
    if "solutions" not in st.session_state:
        st.session_state.solutions = [
            {
                "name": "Privilégier des fournisseurs green",
                "type": "simple",
                "impact_max": 0.8,
                "target": "EF",
                "years_targets": {},
                "categories": {}
            },
            {
                "name": "Réduction des achats",
                "type": "simple",
                "impact_max": 1.0,
                "target": "Value",
                "years_targets": {},
                "categories": {}
            },
            {
                "name": "Prime à l'achat de vélo",
                "type": "simple",
                "impact_max": 1.0,
                "target": "Value",
                "years_targets": {},
                "categories": {}
            }
        ]
        
        
def keep_only_most_specific(paths):
    """
    From a list of hierarchical paths, keep only the most specific ones.
    If a parent and child are both selected, keep only the child.
    """
    sorted_paths = sorted(paths, key=lambda x: len(x), reverse=True)
    kept = []

    for p in sorted_paths:
        if not any(p.startswith(k + " >") or p == k for k in kept):
            kept.append(p)

    return kept


def select_solution(data, years):
    """
    Allow user to configure a solution (simple or mixed) by selecting:
    - Implementation levels per year (after start year)
    - Targeted categories (via tree)
    - Adjusting max impact
    """
    st.subheader("Select an existing solution to configure")

    if "solutions" not in st.session_state or not st.session_state.solutions:
        st.info("No solutions available.")
        return

    selected_name = st.selectbox(
        "Choose a solution",
        [s["name"] for s in st.session_state.solutions]
    )

    selected_solution = next(
        (s for s in st.session_state.solutions if s["name"] == selected_name),
        None
    )

    if selected_solution:
        with st.form(f"form_configure_{selected_name}"):
            st.markdown(f"**Solution**: `{selected_solution['name']}`")
            st.markdown(f"- Type: `{selected_solution['type']}`")
            st.markdown(f"- Target field: `{selected_solution['target']}`")
            st.markdown(f"- Max impact: `{selected_solution['impact_max'] * 100:.1f}%`")

            # Update impact max if needed
            new_impact = st.slider(
                "Maximum impact (theoretical limit)",
                0.0, 1.0,
                selected_solution.get("impact_max", 1.0),
                step=0.05,
                key=f"impact_slider_{selected_name}"
            )
            selected_solution["impact_max"] = new_impact

            # Select start year
            start_year = st.selectbox(
                "Start year of deployment",
                years,
                index=years.index(selected_solution.get("start_year", years[0])) if selected_solution.get("start_year") in years else 0
                )

            selected_solution["start_year"] = start_year

            available_years = [y for y in years if y >= start_year]

            # Set implementation levels
            st.markdown("### Define implementation level per year")
            selected_years = st.multiselect(
                "Select target years",
                available_years,
                default=sorted(int(y) for y in selected_solution.get("years_targets", {}).keys() if int(y) in available_years)
            )

            year_targets = {}
            for y in selected_years:
                pct = st.slider(
                    f"Implementation for {y} (% of max effect)",
                    0, 100,
                    int(selected_solution.get("years_targets", {}).get(str(y), 0) * 100),
                    key=f"{selected_name}_{y}"
                )
                year_targets[y] = pct / 100

            # Assign categories depending on type
            tree = build_tree(data)

            if selected_solution["type"] == "simple":
                st.markdown("### Categories impacted by this solution")
                selection = tree_select(
                        tree,
                        checked=selected_solution.get("categories", {}).get("checked", []),
                        key=f"tree_simple_{selected_name}"
                        )
                selected_solution["categories"] = selection

            elif selected_solution["type"] == "mixed":
                st.markdown("### 📉 Categories to reduce")
                reduction = tree_select(
                    tree,
                    checked=selected_solution.get("reduction", {}).get("categories", {}).get("checked", []),
                    key=f"tree_red_{selected_name}"
                    )
                
                
                
                st.markdown("### 📈 Categories to increase")
                increase = tree_select(
                    tree,
                    checked=selected_solution.get("increase", {}).get("categories", {}).get("checked", []),
                    key=f"tree_inc_{selected_name}"
                    )
                



                factor = st.number_input(
                    "Conversion factor (increase units per reduced unit)",
                    min_value=0.01, format="%.2f",
                    value=selected_solution.get("increase", {}).get("conversion_factor", 1.0)
                )

                selected_solution["reduction"] = {"categories": reduction}
                selected_solution["increase"] = {
                    "categories": increase,
                    "conversion_factor": factor
                }

            submitted = st.form_submit_button("Save configuration")

            if submitted:
                selected_solution["years_targets"] = year_targets
                st.success(f"Configuration for '{selected_solution['name']}' saved.")

def create_solution():
    st.subheader("Create a new solution")

    with st.form("create_solution_form"):
        name = st.text_input("Name of the solution")
        solution_type = st.selectbox("Type of solution", ["simple", "mixed"])
        impact_max = st.slider("Maximum possible impact (0 = no effect, 1 = full effect)", 0.0, 1.0, 0.5)
        target = st.selectbox("Target field", ["EF", "Value"])

        submitted = st.form_submit_button("Add solution")

        if submitted and name:
            new_solution = {
                "name": name,
                "type": solution_type,
                "impact_max": impact_max,
                "target": target,
                "years_targets": {},
                "categories": {}
            }

            # For mixed solutions, prepare increase and reduction placeholders
            if solution_type == "mixed":
                new_solution["reduction"] = {"categories": {}}
                new_solution["increase"] = {"categories": {}, "conversion_factor": 1.0}

            if "solutions" not in st.session_state:
                st.session_state.solutions = []

            st.session_state.solutions.append(new_solution)
            st.success(f"✅ Solution '{name}' of type '{solution_type}' added.")



def interpolate_targets(year_targets, all_years, start_year):
    """
    Interpolate the target values (from year_targets) across all_years.
    Years before `start_year` will be set to 0.

    Parameters:
    - year_targets (dict): {year: proportion} values manually set by the user
    - all_years (List[int]): Full list of years over which to interpolate
    - start_year (int): Year before which no reduction is applied

    Returns:
    - dict: Interpolated proportions per year
    """
    interpolated = {}

    if not year_targets:
        return {y: 0.0 for y in all_years}

    # Convert keys to int in case they were strings
    year_targets_int = {int(k): v for k, v in year_targets.items()}

    # Ensure the interpolation starts at the defined start_year
    if start_year not in year_targets_int:
        year_targets_int[start_year] = 0.0

    sorted_targets = sorted(year_targets_int.items())
    known_years = [y for y, _ in sorted_targets]

    for year in all_years:
        if year < start_year:
            interpolated[year] = 0.0
        elif year in year_targets_int:
            interpolated[year] = year_targets_int[year]
        elif year > known_years[-1]:
            interpolated[year] = year_targets_int[known_years[-1]]
        else:
            for j in range(1, len(known_years)):
                y0, y1 = known_years[j - 1], known_years[j]
                if y0 < year < y1:
                    v0 = year_targets_int[y0]
                    v1 = year_targets_int[y1]
                    ratio = (year - y0) / (y1 - y0)
                    interpolated[year] = v0 + ratio * (v1 - v0)
                    break

    return interpolated





def apply_solutions(df, years):
    """
    Apply both simple and mixed solutions to the projection DataFrame, using full hierarchical path checks.
    """
    if "solutions" not in st.session_state or not st.session_state.solutions:
        return df

    modified_df = df.copy()

    for sol in st.session_state.solutions:
        st.markdown(f"## 🛠️ Applying solution: {sol['name']}")
        st.json(sol)

        impact_max = sol.get("impact_max", 0)
        target_field = sol.get("target", "EF")

        if sol["type"] == "simple":
            raw_targets = sol.get("years_targets", {})
            start_year = sol.get("start_year", years[0])
            interpolated_targets = interpolate_targets(raw_targets, years, start_year)
            selected = set(sol.get("categories", {}).get("checked", []))

            for idx, row in modified_df.iterrows():
                full_label = get_label_path(row)

                if is_subpath(full_label, selected):
                    for year in years:
                        col = f"{target_field}_{year}"
                        if col in modified_df.columns:
                            reduction = impact_max * interpolated_targets.get(year, 0.0)
                            before = modified_df.at[idx, col]
                            after = before * (1 - reduction)
                            modified_df.at[idx, col] = after

                            st.write(f"✅ {full_label} | {col}: {before:.2f} → {after:.2f} (-{reduction*100:.1f}%)")
                else:
                    st.write(f"🔍 Not matched: {full_label}")

        elif sol["type"] == "mixed":
            st.markdown("### 🔍 Mixed solution detected")

            reduction_paths = set(sol.get("reduction", {}).get("categories", {}).get("checked", []))
            increase_paths = set(sol.get("increase", {}).get("categories", {}).get("checked", []))
            factor = sol.get("increase", {}).get("conversion_factor", 1.0)

            raw_targets = sol.get("years_targets", {})
            start_year = sol.get("start_year", years[0])
            interpolated_targets = interpolate_targets(raw_targets, years, start_year)

            st.json({
                "reduction_categories": list(reduction_paths),
                "increase_categories": list(increase_paths),
                "factor": factor,
                "interpolated_targets": interpolated_targets
            })

            yearly_reductions = {y: 0.0 for y in years}

            # Phase 1: Apply reductions
            for idx, row in modified_df.iterrows():
                full_label = get_label_path(row)

                if is_subpath(full_label, reduction_paths):
                    for year in years:
                        col = f"{target_field}_{year}"
                        if col in modified_df.columns:
                            reduction = impact_max * interpolated_targets.get(year, 0.0)
                            before = modified_df.at[idx, col]
                            delta = before * reduction
                            modified_df.at[idx, col] = before - delta
                            yearly_reductions[year] += delta

                            st.write(f"🧊 REDUCE {full_label} | {col}: {before:.2f} → {before - delta:.2f} (-{reduction*100:.1f}%)")

            # Phase 2: Distribute increases
            affected_rows = []
            for idx, row in modified_df.iterrows():
                full_label = get_label_path(row)
                if is_subpath(full_label, increase_paths):
                    affected_rows.append(idx)

            st.write(f"👥 Rows to increase: {len(affected_rows)}")

            if affected_rows:
                for year in years:
                    col = f"{target_field}_{year}"
                    total_increase = yearly_reductions[year] * factor
                    if len(affected_rows) > 0:
                        per_row_increase = total_increase / len(affected_rows)
                        for idx in affected_rows:
                            before = modified_df.at[idx, col]
                            modified_df.at[idx, col] += per_row_increase
                            st.write(f"🚀 INCREASE idx {idx} | {col}: {before:.2f} → {modified_df.at[idx, col]:.2f} (+{per_row_increase:.2f})")

    return modified_df



def is_subpath(path, selected_paths):
    """
    Check if a given path is a subpath or exact match of one of the selected paths.
    Used to determine if a row belongs to a selected category.
    """
    return any(path == sel or path.startswith(sel + " >") for sel in selected_paths)


def get_label_path(row):
    """
    Construct a full hierarchical label from a row: Category > Sub1 > Sub2 > Sub3 > Name > Location
    """
    parts = [
        row.get("Category"),
        row.get("Sub-category 1"),
        row.get("Sub-category 2"),
        row.get("Sub-category 3"),
        row.get("Name"),
        row.get("Location")
    ]
    return " > ".join(str(p).strip() for p in parts if pd.notna(p))

def compute_emissions_per_year(df, years):
    """
    Compute emissions per line and per year: EF × Value.
    Returns a new DataFrame with emissions columns: Emissions_YYYY
    """
    emissions_df = df.copy()
    for y in years:
        emissions_df[f"Emissions_{y}"] = df[f"EF_{y}"] * df[f"Value_{y}"]
    return emissions_df

def compute_avoided_emissions(df_before, df_after, years):
    """
    Compute avoided emissions = Emissions_before - Emissions_after
    for each line and year.
    """
    avoided_df = df_before[[c for c in df_before.columns if "Emissions_" in c]].copy()
    for y in years:
        col = f"Emissions_{y}"
        avoided_df[col] = df_before[col] - df_after[col]
    return avoided_df

def build_solution_weights_table(df, years, st_session_solutions):
    """
    For each row and year, list the solutions that impact EF and/or Value
    with their relative weight (impact_max × implementation level).
    Returns two dicts:
    - ef_weights[row_index][year] = {"Solution A": 0.6, ...}
    - val_weights[row_index][year] = {"Solution B": 1.0, ...}
    """
    ef_weights = {idx: {y: {} for y in years} for idx in df.index}
    val_weights = {idx: {y: {} for y in years} for idx in df.index}

    for sol in st_session_solutions:
        name = sol["name"]
        sol_type = sol["type"]
        sol_target = sol.get("target", "")
        start_year = sol.get("start_year", years[0])
        impact_max = sol.get("impact_max", 0.0)
        interpolated = interpolate_targets(sol.get("years_targets", {}), years, start_year)

        for y in years:
            level = impact_max * interpolated.get(y, 0.0)
            if level == 0:
                continue

            for idx, row in df.iterrows():
                label = get_label_path(row)

                if sol_type == "simple":
                    selected = set(sol.get("categories", {}).get("checked", []))
                    if is_subpath(label, selected):
                        if sol_target == "EF":
                            ef_weights[idx][y][name] = level
                        elif sol_target == "Value":
                            val_weights[idx][y][name] = level

                elif sol_type == "mixed":
                    red_sel = set(sol.get("reduction", {}).get("categories", {}).get("checked", []))
                    if is_subpath(label, red_sel):
                        if sol_target == "EF":
                            ef_weights[idx][y][name] = level
                        elif sol_target == "Value":
                            val_weights[idx][y][name] = level
                    inc_sel = set(sol.get("increase", {}).get("categories", {}).get("checked", []))
                    if is_subpath(label, inc_sel):
                        if sol_target == "EF":
                            ef_weights[idx][y][name] = level
                        elif sol_target == "Value":
                            val_weights[idx][y][name] = level


    return ef_weights, val_weights

def build_diagnostic_weights_table(df, years, ef_weights, val_weights):
    """
    Return a DataFrame with rows = "idx - EF" or "idx - Value",
    columns = years, and values = list of (solution_name, %)
    """
    diagnostic_rows = []

    for idx in df.index:
        for field in ["EF", "Value"]:
            row_label = f"{idx} - {field}"
            row_data = {}
            for y in years:
                weights = ef_weights[idx][y] if field == "EF" else val_weights[idx][y]
                row_data[y] = [(name, round(100 * w, 1)) for name, w in weights.items()]
            diagnostic_rows.append((row_label, row_data))

    return pd.DataFrame(
        [r[1] for r in diagnostic_rows],
        index=[r[0] for r in diagnostic_rows]
    )


def compute_solution_impact_from_diagnostic(df_before, df_after, df_avoided, diagnostic_df, years):
    """
    Distribute real emission reductions (from df_avoided) to each solution
    based on weights found in diagnostic_df and impact geometry (brut EF/Value).
    
    Prints debug info for idx=0 and year=2026.
    """
    impact_by_solution = {}

    for idx in df_before.index:
        for year in years:
            ef_col = f"EF_{year}"
            val_col = f"Value_{year}"
            em_col = f"Emissions_{year}"

            ef_b = df_before.at[idx, ef_col]
            ef_a = df_after.at[idx, ef_col]
            val_b = df_before.at[idx, val_col]
            val_a = df_after.at[idx, val_col]

            delta = df_avoided.at[idx, em_col]
            if delta == 0:
                continue

            key_ef = f"{idx} - EF"
            key_val = f"{idx} - Value"
            ef_weights = diagnostic_df.loc[key_ef, year] if key_ef in diagnostic_df.index else []
            val_weights = diagnostic_df.loc[key_val, year] if key_val in diagnostic_df.index else []

            ef_dict = {s: pct / 100 for s, pct in ef_weights} if isinstance(ef_weights, list) else {}
            val_dict = {s: pct / 100 for s, pct in val_weights} if isinstance(val_weights, list) else {}

            brut_ef = (ef_b - ef_a) * val_b 
            brut_val = (val_b - val_a) * ef_b 
            brut_total = brut_ef + brut_val

            # 🔍 Debug pour ligne 0 et année 2026
            if idx == 0 and year == 2026:
                st.markdown(f"### 🧪 DEBUG — Ligne {idx} | Année {year}")
                st.write(f"EF_before = {ef_b}, EF_after = {ef_a}")
                st.write(f"Value_before = {val_b}, Value_after = {val_a}")
                st.write(f"brut_ef = {brut_ef:.4f}, brut_val = {brut_val:.4f}, brut_total = {brut_total:.4f}, delta = {delta:.4f}")
                st.write(f"EF weights = {ef_dict}")
                st.write(f"Value weights = {val_dict}")

            if brut_total == 0:
                continue

            # EF-based attribution
            total_ef_weight = sum(ef_dict.values())
            for sol, w in ef_dict.items():
                share = w / total_ef_weight if total_ef_weight else 0
                real_impact = share * (brut_ef / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

                if idx == 0 and year == 2026:
                    st.markdown("**EF → Attribution**")
                    st.write(f"{sol}: poids = {w:.4f}, part = {share:.2%}, impact réel = {real_impact:.4f}")

            # Value-based attribution
            total_val_weight = sum(val_dict.values())
            for sol, w in val_dict.items():
                share = w / total_val_weight if total_val_weight else 0
                real_impact = share * (brut_val / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

                if idx == 0 and year == 2026:
                    st.markdown("**Value → Attribution**")
                    st.write(f"{sol}: poids = {w:.4f}, part = {share:.2%}, impact réel = {real_impact:.4f}")

    final = pd.DataFrame.from_dict(impact_by_solution, orient="index").fillna(0.0)
    final = final[[y for y in years if y in final.columns]]
    final.index.name = "Solution"
    return final







