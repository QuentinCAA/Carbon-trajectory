# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 08:21:30 2025

@author: quent
"""

import streamlit as st
from modules.tree import build_tree
from streamlit_tree_select import tree_select
import pandas as pd

def init_solutions():
    """
    Initialise the default list of mitigation solutions in session state if not already present.

    Each solution defines an action that reduces emissions either by lowering the emission factor (EF)
    or by reducing the activity value. These default entries are of type 'simple', and can later
    be configured with year-specific impacts and category assignments.

    Effects:
    - Adds a 'solutions' key to st.session_state if it does not exist.
    - Populates it with a predefined list of example solutions, each containing:
        - 'name': Description of the solution.
        - 'type': Solution type ('simple' by default).
        - 'impact_max': Maximum reduction ratio (e.g. 0.8 = 20% reduction).
        - 'target': Whether the solution reduces 'EF' or 'Value'.
        - 'years_targets': Dictionary of year-to-impact values (initially empty).
        - 'categories': Dictionary for category tree selections (initially empty).
    """
    
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
    Filter a list of hierarchical paths to keep only the most specific entries.

    If both a parent and one of its descendants are present in the list,
    only the most specific (i.e. the deepest) path is retained. This avoids 
    duplicate or overlapping application of actions on hierarchical trees.

    Parameters:
    - paths (List[str]): List of hierarchical paths, e.g. "Category > Sub-category > Name".

    Returns:
    - List[str]: Filtered list containing only the most specific (non-redundant) paths.
    """
    sorted_paths = sorted(paths, key=lambda x: len(x), reverse=True)
    kept = []

    for p in sorted_paths:
        if not any(p.startswith(k + " >") or p == k for k in kept):
            kept.append(p)

    return kept

def create_solution():
    """
    Display a form to create a new mitigation solution and store it in session state.
    
    Users can define a solution by setting its name, type (simple or mixed), target field,
    and maximum possible impact. For mixed solutions, additional placeholders are initialized
    for reduction and increase configurations.
    
    Effects:
    - Displays a form for entering solution details.
    - Appends the new solution to st.session_state["solutions"] with:
        - 'name': Name of the solution.
        - 'type': "simple" or "mixed".
        - 'impact_max': Theoretical maximum effect (between 0.0 and 1.0).
        - 'target': Target field ("EF" or "Value").
        - 'years_targets': Empty dictionary for future yearly implementation levels.
        - 'categories': Empty for simple solutions.
        - 'reduction' and 'increase': Initialized for mixed solutions.
    - Shows a success message when the solution is successfully added.
    """
    
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


def select_solution(data, years):
    """
    Configure an existing mitigation solution (simple or mixed) via an interactive form.
    
    This function allows the user to:
    - Adjust the maximum impact of the solution.
    - Define the start year and implementation levels per year.
    - Assign target categories via tree selection.
    - (For mixed solutions) Define both reduced and increased categories,
      along with a conversion factor.
    
    Parameters:
    - data (pd.DataFrame): Dataset used to build the hierarchical tree of categories.
    - years (List[int]): List of available projection years.
    
    Effects:
    - Updates the selected solution in st.session_state["solutions"] with:
        - 'impact_max': Maximum possible reduction ratio.
        - 'start_year': Year the solution begins implementation.
        - 'years_targets': Mapping of year to % of effect applied.
        - 'categories': For simple solutions, the affected categories.
        - 'reduction' and 'increase': For mixed solutions, the respective trees and conversion factor.
    - Displays a success message upon saving the configuration.
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
            
            # Unique prefix to differentiate keys across solutions
            solution_key_prefix = f"impl_{selected_name}"
            
            year_targets = {}
            
            for y in selected_years:
                key = f"{solution_key_prefix}_{y}"
                default_pct = int(selected_solution.get("years_targets", {}).get(str(y), 0) * 100)
            
                # Ensure session_state is initialized
                if key not in st.session_state:
                    st.session_state[key] = default_pct
            
                pct = st.slider(
                    f"Implementation for {y} (% of max effect)",
                    0, 100,
                    key=key
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



def interpolate_targets(year_targets, all_years, start_year):
    """
     Interpolate target values across all years based on manually defined target points.
    
     This function fills in missing years between defined target values using linear interpolation.
     Years before the start year are set to 0. For years after the last known target, the final
     value is extended. This ensures a complete mapping of year → effect level for each solution.
    
     Parameters:
     - year_targets (dict): Dictionary of user-defined proportions per year (e.g. {"2026": 0.3, "2028": 0.7}).
     - all_years (List[int]): List of all years to cover (e.g. [2025, ..., 2035]).
     - start_year (int): Year before which all proportions should be 0.0.
    
     Returns:
     - dict: A dictionary mapping each year to its interpolated proportion (float between 0 and 1).
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
    Apply all configured mitigation solutions (simple and mixed) to the projection DataFrame.

    This function updates the projection DataFrame in two phases:
    - For simple solutions: it applies a proportional reduction to either EF or Value,
      based on user-defined target levels and selected categories.
    - For mixed solutions: it first applies reductions to one group of categories,
      then redistributes the saved emissions proportionally to another group, using a conversion factor.

    Parameters:
    - df (pd.DataFrame): Projection DataFrame containing 'Value_YEAR' and/or 'EF_YEAR' columns.
    - years (List[int]): List of years over which the solutions are applied.

    Returns:
    - pd.DataFrame: A new DataFrame with updated values after applying all solutions.
    """
    
    if "solutions" not in st.session_state or not st.session_state.solutions:
        return df

    modified_df = df.copy()

    for sol in st.session_state.solutions:
        #st.markdown(f"#### 🛠️ Applying solution: {sol['name']}")
        #st.json(sol)

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

                            #st.write(f"✅ {full_label} | {col}: {before:.2f} → {after:.2f} (-{reduction*100:.1f}%)")
                else:
                    continue
                    #st.write(f"🔍 Not matched: {full_label}")

        elif sol["type"] == "mixed":
            # st.markdown("### 🔍 Mixed solution detected")

            reduction_paths = set(sol.get("reduction", {}).get("categories", {}).get("checked", []))
            increase_paths = set(sol.get("increase", {}).get("categories", {}).get("checked", []))
            factor = sol.get("increase", {}).get("conversion_factor", 1.0)

            raw_targets = sol.get("years_targets", {})
            start_year = sol.get("start_year", years[0])
            interpolated_targets = interpolate_targets(raw_targets, years, start_year)

            # st.json({"reduction_categories": list(reduction_paths),"increase_categories": list(increase_paths),"factor": factor,"interpolated_targets": interpolated_targets})

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

                            # st.write(f"🧊 REDUCE {full_label} | {col}: {before:.2f} → {before - delta:.2f} (-{reduction*100:.1f}%)")

            # Phase 2: Distribute increases
            affected_rows = []
            for idx, row in modified_df.iterrows():
                full_label = get_label_path(row)
                if is_subpath(full_label, increase_paths):
                    affected_rows.append(idx)

            # st.write(f"👥 Rows to increase: {len(affected_rows)}")

            if affected_rows:
                for year in years:
                    col = f"{target_field}_{year}"
                    total_increase = yearly_reductions[year] * factor
                    if len(affected_rows) > 0:
                        per_row_increase = total_increase / len(affected_rows)
                        for idx in affected_rows:
                            before = modified_df.at[idx, col]
                            modified_df.at[idx, col] += per_row_increase
                            # st.write(f"🚀 INCREASE idx {idx} | {col}: {before:.2f} → {modified_df.at[idx, col]:.2f} (+{per_row_increase:.2f})")

    return modified_df



def is_subpath(path, selected_paths):
    """
    Check whether a given hierarchical path is a subpath or exact match of any selected path.
    
    This is used to determine whether an element (e.g. row in the emissions table)
    is included under one of the selected categories, considering full hierarchy.
    
    Parameters:
    - path (str): The full path to check, e.g. "Category > Sub1 > Name".
    - selected_paths (Iterable[str]): List or set of selected reference paths.
    
    Returns:
    - bool: True if path matches or is nested under one of the selected paths.
    """
    return any(path == sel or path.startswith(sel + " >") for sel in selected_paths)


def get_label_path(row):
    """
    Construct a full hierarchical label from a DataFrame row by concatenating non-empty levels.

    The resulting label includes up to six levels:
    Category > Sub-category 1 > Sub-category 2 > Sub-category 3 > Name > Location.

    Parameters:
    - row (pd.Series): A row from the emissions DataFrame.

    Returns:
    - str: A string representing the full hierarchical path of the row.
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
    Compute annual emissions per row by multiplying EF and Value columns.

    For each year, this function calculates:
        Emissions_YEAR = EF_YEAR × Value_YEAR

    Parameters:
    - df (pd.DataFrame): DataFrame containing EF_YEAR and Value_YEAR columns.
    - years (List[int]): List of years over which to compute emissions.

    Returns:
    - pd.DataFrame: A new DataFrame with additional Emissions_YEAR columns.
    """
    emissions_df = df.copy()
    for y in years:
        emissions_df[f"Emissions_{y}"] = df[f"EF_{y}"] * df[f"Value_{y}"]
    return emissions_df


def compute_avoided_emissions(df_before, df_after, years):
    """
    Compute avoided emissions per row and per year by comparing before/after values.

    This function calculates, for each year:
        Avoided = Emissions_before - Emissions_after

    Parameters:
    - df_before (pd.DataFrame): DataFrame with original emissions (must include Emissions_YEAR columns).
    - df_after (pd.DataFrame): DataFrame with emissions after solutions are applied.
    - years (List[int]): List of years to include in the calculation.

    Returns:
    - pd.DataFrame: DataFrame with Emissions_YEAR columns representing avoided emissions.
    """
    avoided_df = df_before[[c for c in df_before.columns if "Emissions_" in c]].copy()
    for y in years:
        col = f"Emissions_{y}"
        avoided_df[col] = df_before[col] - df_after[col]
    return avoided_df

def build_solution_weights_table(df, years, st_session_solutions):
    """
    Build internal weight tables showing how each solution contributes to each row and year.

    For each row and year, this function determines which solutions affect the row
    (based on category matching) and stores the weight of their effect
    as: impact_max × implementation level (from interpolation).

    Parameters:
    - df (pd.DataFrame): Projection DataFrame with one row per item.
    - years (List[int]): List of projection years.
    - st_session_solutions (List[dict]): List of configured solutions from session state.

    Returns:
    - Tuple[dict, dict]: Two nested dictionaries:
        - ef_weights[row_idx][year] = {solution_name: weight}
        - val_weights[row_idx][year] = {solution_name: weight}
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
    Build a diagnostic DataFrame showing solution weights per row and year.

    Each row of the output corresponds to either EF or Value weights for a given index.
    The values are lists of (solution_name, weight%) for each year.

    Parameters:
    - df (pd.DataFrame): Original projection DataFrame.
    - years (List[int]): List of projection years.
    - ef_weights (dict): Weight contributions to EF per row and year.
    - val_weights (dict): Weight contributions to Value per row and year.

    Returns:
    - pd.DataFrame: Diagnostic DataFrame with rows like "0 - EF", "0 - Value"
      and columns as years containing lists of (solution, weight%) tuples.
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
    Attribute the real avoided emissions to each solution using diagnostic weight tables.

    Based on the proportional weights applied to EF and Value per solution,
    this function distributes the actual avoided emissions (computed before - after)
    across all active solutions, considering their relative contribution and
    the geometry of the change (EF vs. Value).

    Parameters:
    - df_before (pd.DataFrame): Emissions DataFrame before applying solutions.
    - df_after (pd.DataFrame): Emissions DataFrame after applying solutions.
    - df_avoided (pd.DataFrame): Emissions_YYYY difference between before and after.
    - diagnostic_df (pd.DataFrame): Diagnostic weight table with EF and Value attribution per row.
    - years (List[int]): List of projection years.

    Returns:
    - pd.DataFrame: Final attribution table with one row per solution and one column per year,
      showing the amount of emissions (in absolute units) avoided due to each solution.
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

            # 🔍 Possible to display to debug by choosing a lign and a year
            # if idx == 0 and year == 2026:
                # st.markdown(f"### 🧪 DEBUG — Ligne {idx} | Année {year}")
                # st.write(f"EF_before = {ef_b}, EF_after = {ef_a}")
                # st.write(f"Value_before = {val_b}, Value_after = {val_a}")
                # st.write(f"brut_ef = {brut_ef:.4f}, brut_val = {brut_val:.4f}, brut_total = {brut_total:.4f}, delta = {delta:.4f}")
                # st.write(f"EF weights = {ef_dict}")
                # st.write(f"Value weights = {val_dict}")

            if brut_total == 0:
                continue

            # EF-based attribution
            total_ef_weight = sum(ef_dict.values())
            for sol, w in ef_dict.items():
                share = w / total_ef_weight if total_ef_weight else 0
                real_impact = share * (brut_ef / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

                # if idx == 0 and year == 2026:
                    # st.markdown("**EF → Attribution**")
                    # st.write(f"{sol}: poids = {w:.4f}, part = {share:.2%}, impact réel = {real_impact:.4f}")

            # Value-based attribution
            total_val_weight = sum(val_dict.values())
            for sol, w in val_dict.items():
                share = w / total_val_weight if total_val_weight else 0
                real_impact = share * (brut_val / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

                # if idx == 0 and year == 2026:
                    # st.markdown("**Value → Attribution**")
                    # st.write(f"{sol}: poids = {w:.4f}, part = {share:.2%}, impact réel = {real_impact:.4f}")

    final = pd.DataFrame.from_dict(impact_by_solution, orient="index").fillna(0.0)
    final = final[[y for y in years if y in final.columns]]
    final.index.name = "Solution"
    return final

