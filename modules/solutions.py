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
        

def create_solution():
    """
    Display a form to create a new mitigation solution and store it in session state.
    
    Users can define a solution by setting its name, type (simple or mixed), target field,
    and maximum possible impact. For mixed solutions, additional placeholders are initialized
    for reduction and increase configurations.
    
    Effects:
    - Displays a form for entering solution details.
    - Appends the new solution to st.session_state["solutions"].
    - Also registers the new entry in st.session_state["solutions_table"].
    - Shows a success message upon creation.
    """
    import streamlit as st

    st.subheader("➕ Create a new solution")

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

            # For mixed solutions, add placeholders
            if solution_type == "mixed":
                new_solution["reduction"] = {"categories": {}}
                new_solution["increase"] = {"categories": {}, "conversion_factor": 1.0}

            # Ensure the list exists and append
            if "solutions" not in st.session_state:
                st.session_state.solutions = []

            st.session_state.solutions.append(new_solution)

            # ✅ Register in central table for yearly targets
            if "solutions_table" not in st.session_state:
                st.session_state.solutions_table = {}
            st.session_state.solutions_table[name] = {}

            st.success(f"✅ Solution '{name}' ({solution_type}) created successfully.")




def select_solution(data, years):
    """
    Configure all mitigation solutions (simple or mixed) via independent forms.
    
    Each solution has its own form. Updates are written directly to
    st.session_state.solutions[i] without rebuilding any lists or external tables.
    """

    import streamlit as st
    import re

    st.subheader("⚙️ Configure existing solutions")

    # Safety check
    if "solutions" not in st.session_state or not st.session_state.solutions:
        st.info("No solutions available yet. Please create one first.")
        return

    # Build hierarchical tree of categories
    tree = build_tree(data)
    cols = st.columns(3)

    # One form per solution
    for i, sol in enumerate(st.session_state.solutions):
        form_id = re.sub(r"\W+", "_", sol["name"])
        col = cols[i % 3]

        with col.form(f"form_edit_solution_{form_id}"):
            st.markdown(f"### 💡 `{sol['name']}`")
            st.markdown(f"- Type: `{sol['type']}` | Target: `{sol['target']}`")

            # --- Maximum impact
            new_impact = st.slider(
                "Maximum impact (theoretical limit)",
                0.0, 1.0,
                sol.get("impact_max", 1.0),
                step=0.05,
                key=f"impact_{sol['name']}"
            )
            st.session_state.solutions[i]["impact_max"] = new_impact

            # --- Start year
            start_year = st.selectbox(
                "Start year",
                years,
                index=years.index(sol.get("start_year", years[0]))
                if sol.get("start_year") in years else 0,
                key=f"start_{sol['name']}"
            )
            st.session_state.solutions[i]["start_year"] = start_year

            # --- Implementation per year
            available_years = [y for y in years if y >= start_year]
            st.markdown("### Implementation level per year")

            year_targets = sol.get("years_targets", {})
            local_targets = {}

            selected_years = st.multiselect(
                "Select target years",
                available_years,
                default=sorted(int(y) for y in year_targets.keys()),
                key=f"years_{sol['name']}"
            )

            for y in selected_years:
                pct = st.slider(
                    f"{y} (% of max effect)",
                    0, 100,
                    int(year_targets.get(str(y), 0) * 100),
                    key=f"{sol['name']}_impl_{y}"
                )
                local_targets[str(y)] = pct / 100.0

            st.session_state.solutions[i]["years_targets"] = local_targets

            # --- Category trees
            if sol["type"] == "simple":
                st.markdown("### Categories impacted by this solution")
                selection = tree_select(
                    tree,
                    checked=sol.get("categories", {}).get("checked", []),
                    expanded=sol.get("categories", {}).get("expanded", []),
                    key=f"tree_simple_{sol['name']}"
                )
                st.session_state.solutions[i]["categories"] = selection

            elif sol["type"] == "mixed":
                st.markdown("### 📉 Categories to reduce")
                reduction = tree_select(
                    tree,
                    checked=sol.get("reduction", {}).get("categories", {}).get("checked", []),
                    expanded=sol.get("reduction", {}).get("categories", {}).get("expanded", []),
                    key=f"tree_red_{sol['name']}"
                )

                st.markdown("### 📈 Categories to increase")
                increase = tree_select(
                    tree,
                    checked=sol.get("increase", {}).get("categories", {}).get("checked", []),
                    expanded=sol.get("increase", {}).get("categories", {}).get("expanded", []),
                    key=f"tree_inc_{sol['name']}"
                )

                factor = st.number_input(
                    "Conversion factor (increase units per reduced unit)",
                    min_value=0.01,
                    format="%.2f",
                    value=sol.get("increase", {}).get("conversion_factor", 1.0),
                    key=f"factor_{sol['name']}"
                )

                st.session_state.solutions[i]["reduction"] = {"categories": reduction}
                st.session_state.solutions[i]["increase"] = {
                    "categories": increase,
                    "conversion_factor": factor
                }

            # --- Save button (does not rebuild, just confirms)
            submitted = st.form_submit_button("Save configuration")
            if submitted:
                st.success(f"✅ Configuration for '{sol['name']}' saved.")


    



def apply_solutions(df, years):
    """
    Apply all configured mitigation solutions (simple and mixed) to the projection DataFrame.

    Uses both 'impact_max' and 'years_targets' to compute the proportional effect
    on EF or Value fields.

    Parameters:
    - df (pd.DataFrame): Projection DataFrame with 'Value_YEAR' and 'EF_YEAR' columns.
    - years (List[int]): Projection years.

    Returns:
    - pd.DataFrame: Updated DataFrame with solutions applied.
    """
    if "solutions" not in st.session_state or not st.session_state.solutions:
        return df

    modified_df = df.copy()

    for sol in st.session_state.solutions:
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
                            modified_df.at[idx, col] = before * (1 - reduction)

        elif sol["type"] == "mixed":
            reduction_paths = set(sol.get("reduction", {}).get("categories", {}).get("checked", []))
            increase_paths = set(sol.get("increase", {}).get("categories", {}).get("checked", []))
            factor = sol.get("increase", {}).get("conversion_factor", 1.0)

            raw_targets = sol.get("years_targets", {})
            start_year = sol.get("start_year", years[0])
            interpolated_targets = interpolate_targets(raw_targets, years, start_year)
            yearly_reductions = {y: 0.0 for y in years}

            # --- Phase 1: apply reductions
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

            # --- Phase 2: redistribute increases
            affected_rows = [
                idx for idx, row in modified_df.iterrows()
                if is_subpath(get_label_path(row), increase_paths)
            ]

            for year in years:
                col = f"{target_field}_{year}"
                total_increase = yearly_reductions[year] * factor
                if affected_rows:
                    per_row_increase = total_increase / len(affected_rows)
                    for idx in affected_rows:
                        modified_df.at[idx, col] += per_row_increase

    return modified_df



def build_solution_weights_table(df, years, st_session_solutions):
    """
    Build weight tables showing how each solution contributes to each row and year.

    The weight for each solution = impact_max × interpolated(year_target).
    These weights are later used for detailed emission attribution.

    Parameters:
    - df (pd.DataFrame): Projection DataFrame.
    - years (List[int]): Projection years.
    - st_session_solutions (List[dict]): Configured solutions from session state.

    Returns:
    - Tuple[dict, dict]: (ef_weights, val_weights)
    """
    ef_weights = {idx: {y: {} for y in years} for idx in df.index}
    val_weights = {idx: {y: {} for y in years} for idx in df.index}

    for sol in st_session_solutions:
        name = sol["name"]
        sol_type = sol["type"]
        sol_target = sol.get("target", "")
        impact_max = sol.get("impact_max", 0.0)
        start_year = sol.get("start_year", years[0])
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

