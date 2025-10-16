# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 08:21:30 2025

@author: quent
"""

import streamlit as st
from modules.tree import build_tree
from streamlit_tree_select import tree_select
import pandas as pd


# =========================================================
# 1️⃣ INITIALIZATION
# =========================================================
def init_solutions():
    """
    Initialise the default list of mitigation solutions in session state if not already present.

    Each solution defines an action that reduces emissions either by lowering the emission factor (EF)
    or by reducing the activity value. These default entries are of type 'simple', and can later
    be configured with year-specific impacts and category assignments.
    """
    if "solutions" not in st.session_state:
        st.session_state.solutions = [
            {
                "name": "Green procurement policy",
                "type": "simple",
                "decarbonation_potential": 0.2,
                "target": "EF",
                "years_targets": {},
                "categories": {}
            },
            {
                "name": "Reduced purchasing volumes",
                "type": "simple",
                "decarbonation_potential": 0.25,
                "target": "Value",
                "years_targets": {},
                "categories": {}
            },
            {
                "name": "Bike purchase incentive",
                "type": "simple",
                "decarbonation_potential": 0.3,
                "target": "Value",
                "years_targets": {},
                "categories": {}
            }
        ]


# =========================================================
# 2️⃣ CREATION
# =========================================================
def create_solution():
    """
    Create a new mitigation solution and store it in session state.

    - Uses 'Decarbonation potential (%)' instead of 'impact_max'
    - Numeric inputs (stable across sessions)
    """

    st.subheader("➕ Create a new solution")

    st.markdown(
        """
        <div style='color: grey; font-size: 0.9em;'>
        💡 Define your new mitigation solution.<br>
        Set its <strong>Decarbonation potential</strong> (maximum theoretical effect, in %) 
        and select the target field it affects.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("create_solution_form"):
        name = st.text_input("Name of the solution")
        solution_type = st.selectbox("Type of solution", ["simple", "mixed"])
        target = st.selectbox("Target field", ["EF", "Value"])

        decarb_potential = st.number_input(
            "Decarbonation potential (%) — e.g. 20 = 20% max reduction",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            format="%.2f"
        )

        submitted = st.form_submit_button("Add solution")

        if submitted and name:
            new_solution = {
                "name": name,
                "type": solution_type,
                "decarbonation_potential": decarb_potential / 100.0,  # store as ratio
                "target": target,
                "years_targets": {},
                "categories": {},
            }

            if solution_type == "mixed":
                new_solution["reduction"] = {"categories": {}}
                new_solution["increase"] = {"categories": {}, "conversion_factor": 1.0}

            if "solutions" not in st.session_state:
                st.session_state.solutions = []

            st.session_state.solutions.append(new_solution)
            st.success(f"✅ Solution '{name}' ({solution_type}) created successfully.")


# =========================================================
# 3️⃣ CONFIGURATION / ASSIGNMENT
# =========================================================
def select_solution(data, years):
    """
    Configure, rename, reorder, or delete existing mitigation solutions.
    Fixed version for Streamlit Cloud: forms have a submit button to avoid warnings,
    and Move/Delete buttons are handled outside of forms for stability.
    """
    import re

    st.subheader("⚙️ Configure existing solutions")

    st.markdown(
        """
        <div style='color: grey; font-size: 0.9em;'>
        💡 You can rename, reorder, or delete solutions directly here.<br>
        - Use <strong>Decarbonation potential</strong> to set the maximum theoretical reduction (in %).<br>
        - For <strong>mixed solutions</strong>, you can now define several increase groups.<br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Safety check
    if "solutions" not in st.session_state or not st.session_state.solutions:
        st.info("No solutions available yet. Please create one first.")
        return

    tree = build_tree(data)
    cols = st.columns(3)
    to_delete = []

    for i, sol in enumerate(st.session_state.solutions):
        form_id = re.sub(r"\W+", "_", sol["name"])
        col = cols[i % 3]

        with col.form(f"form_edit_solution_{form_id}"):
            st.markdown(f"### 💡 Solution {i+1}")

            # --- Editable fields
            new_name = st.text_input("Solution name", value=sol["name"], key=f"name_{i}")
            st.session_state.solutions[i]["name"] = new_name

            st.markdown(f"- Type: `{sol['type']}` | Target: `{sol['target']}`")

            decarb_potential = st.number_input(
                "Decarbonation potential (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(sol.get("decarbonation_potential", 0.0) * 100),
                format="%.2f",
                key=f"potential_{i}"
            )
            st.session_state.solutions[i]["decarbonation_potential"] = decarb_potential / 100.0

            start_year = st.selectbox(
                "Start year",
                years,
                index=years.index(sol.get("start_year", years[0])) if sol.get("start_year") in years else 0,
                key=f"start_{i}"
            )
            st.session_state.solutions[i]["start_year"] = start_year

            # --- Year targets
            available_years = [y for y in years if y >= start_year]
            year_targets = sol.get("years_targets", {})
            local_targets = {}

            selected_years = st.multiselect(
                "Select target years",
                available_years,
                default=sorted(int(y) for y in year_targets.keys()),
                key=f"years_{i}"
            )

            for y in selected_years:
                pct = st.number_input(
                    f"{y} (% of max effect)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(year_targets.get(str(y), 0.0) * 100),
                    format="%.2f",
                    key=f"{i}_impl_{y}"
                )
                local_targets[str(y)] = pct / 100.0

            st.session_state.solutions[i]["years_targets"] = local_targets

            # --- Category tree
            if sol["type"] == "simple":
                st.markdown("### Categories impacted by this solution")
                selection = tree_select(
                    tree,
                    checked=sol.get("categories", {}).get("checked", []),
                    expanded=sol.get("categories", {}).get("expanded", []),
                    key=f"tree_simple_{i}"
                )
                st.session_state.solutions[i]["categories"] = selection

            elif sol["type"] == "mixed":
                st.markdown("### 📉 Categories to reduce")
                reduction = tree_select(
                    tree,
                    checked=sol.get("reduction", {}).get("categories", {}).get("checked", []),
                    expanded=sol.get("reduction", {}).get("categories", {}).get("expanded", []),
                    key=f"tree_red_{i}"
                )
                st.session_state.solutions[i]["reduction"] = {"categories": reduction}

                st.markdown("### 📈 Categories to increase")
                if not isinstance(sol.get("increase"), list):
                    st.session_state.solutions[i]["increase"] = []

                updated_increase_groups = []
                for j, inc in enumerate(sol.get("increase", [])):
                    st.markdown(f"#### ➕ Increase group {j+1}")
                    label = st.text_input(
                        "Label",
                        value=inc.get("label", f"Increase {j+1}"),
                        key=f"inc_label_{i}_{j}"
                    )
                    factor = st.number_input(
                        "Conversion factor",
                        min_value=0.01,
                        format="%.2f",
                        value=float(inc.get("conversion_factor", 1.0)),
                        key=f"factor_{i}_{j}"
                    )
                    inc_selection = tree_select(
                        tree,
                        checked=inc.get("categories", {}).get("checked", []),
                        expanded=inc.get("categories", {}).get("expanded", []),
                        key=f"tree_inc_{i}_{j}"
                    )

                    if not st.checkbox(f"❌ Remove this increase group", key=f"remove_inc_{i}_{j}"):
                        updated_increase_groups.append({
                            "label": label,
                            "categories": inc_selection,
                            "conversion_factor": factor
                        })

                st.session_state.solutions[i]["increase"] = updated_increase_groups

            # ✅ Submit button just to validate the form (avoid “Missing Submit Button”)
            submitted = st.form_submit_button("💾 Save configuration")
            if submitted:
                st.success(f"✅ Configuration for '{new_name}' saved.")

        # === Action buttons OUTSIDE the form ===
        bcol1, bcol2, bcol3 = col.columns([1, 1, 1])
        if bcol1.button("⬆️ Move up", key=f"move_up_{i}") and i > 0:
            st.session_state.solutions[i - 1], st.session_state.solutions[i] = (
                st.session_state.solutions[i],
                st.session_state.solutions[i - 1],
            )
            st.rerun()

        if bcol2.button("⬇️ Move down", key=f"move_down_{i}") and i < len(st.session_state.solutions) - 1:
            st.session_state.solutions[i + 1], st.session_state.solutions[i] = (
                st.session_state.solutions[i],
                st.session_state.solutions[i + 1],
            )
            st.rerun()

        if bcol3.button("🗑️ Delete", key=f"delete_{i}"):
            to_delete.append(i)

    # --- Handle deletions after loop
    if to_delete:
        for idx in sorted(to_delete, reverse=True):
            deleted_name = st.session_state.solutions[idx]["name"]
            del st.session_state.solutions[idx]
            st.warning(f"🗑️ Solution '{deleted_name}' deleted.")


# =========================================================
# 4️⃣ APPLICATION TO DATA
# =========================================================


def apply_solutions(df, years):
    """
    Apply all configured mitigation solutions (simple and mixed) to the projection DataFrame.
    Supports multiple increase groups with their own conversion factors.
    """
    if "solutions" not in st.session_state or not st.session_state.solutions:
        return df

    modified_df = df.copy()
    st.markdown("### 🧪 Interpolation debug options")
    show_debug = st.checkbox("Show interpolation details for each solution", value=False)

    for sol in st.session_state.solutions:
        name = sol["name"]
        potential = sol.get("decarbonation_potential", 0)
        target_field = sol.get("target", "EF")

        st.markdown(f"#### 🔍 Processing solution: **{name}**")

        if sol["type"] == "simple":
            raw_targets = sol.get("years_targets", {})
            start_year = sol.get("start_year", years[0])
            interpolated_targets = interpolate_targets(raw_targets, years, start_year, show_debug=show_debug)
            selected = set(sol.get("categories", {}).get("checked", []))

            for idx, row in modified_df.iterrows():
                full_label = get_label_path(row)
                if is_subpath(full_label, selected):
                    for year in years:
                        col = f"{target_field}_{year}"
                        if col in modified_df.columns:
                            reduction = potential * interpolated_targets.get(year, 0.0)
                            modified_df.at[idx, col] *= (1 - reduction)

        elif sol["type"] == "mixed":
            reduction_paths = set(sol.get("reduction", {}).get("categories", {}).get("checked", []))
            increase_groups = sol.get("increase", [])
            raw_targets = sol.get("years_targets", {})
            start_year = sol.get("start_year", years[0])
            interpolated_targets = interpolate_targets(raw_targets, years, start_year, show_debug=show_debug)
            yearly_reductions = {y: 0.0 for y in years}

            # Phase 1 — Apply reductions
            for idx, row in modified_df.iterrows():
                full_label = get_label_path(row)
                if is_subpath(full_label, reduction_paths):
                    for year in years:
                        col = f"{target_field}_{year}"
                        if col in modified_df.columns:
                            reduction = potential * interpolated_targets.get(year, 0.0)
                            before = modified_df.at[idx, col]
                            delta = before * reduction
                            modified_df.at[idx, col] = before - delta
                            yearly_reductions[year] += delta

            # Phase 2 — Redistribute across all increase groups
            for inc_group in increase_groups:
                factor = inc_group.get("conversion_factor", 1.0)
                increase_paths = set(inc_group.get("categories", {}).get("checked", []))

                affected_rows = [
                    idx for idx, row in modified_df.iterrows()
                    if is_subpath(get_label_path(row), increase_paths)
                ]

                for year in years:
                    col = f"{target_field}_{year}"
                    total_increase = yearly_reductions[year] * factor
                    if affected_rows:
                        per_row_increase = total_increase 
                        for idx in affected_rows:
                            modified_df.at[idx, col] += per_row_increase

        st.markdown("---")

    return modified_df




def build_solution_weights_table(df, years, st_session_solutions):
    """
    Build weight tables showing how each solution contributes to each row and year.

    The weight for each solution = decarbonation_potential × interpolated(year_target).
    These weights are later used for detailed emission attribution.

    Compatible with multiple increase groups (list) in mixed solutions.
    """
    ef_weights = {idx: {y: {} for y in years} for idx in df.index}
    val_weights = {idx: {y: {} for y in years} for idx in df.index}

    for sol in st_session_solutions:
        name = sol["name"]
        sol_type = sol["type"]
        sol_target = sol.get("target", "")
        potential = sol.get("decarbonation_potential", 0.0)
        start_year = sol.get("start_year", years[0])
        interpolated = interpolate_targets(sol.get("years_targets", {}), years, start_year)

        for y in years:
            level = potential * interpolated.get(y, 0.0)
            if level == 0:
                continue

            for idx, row in df.iterrows():
                label = get_label_path(row)

                # === SIMPLE SOLUTIONS ===
                if sol_type == "simple":
                    selected = set(sol.get("categories", {}).get("checked", []))
                    if is_subpath(label, selected):
                        if sol_target == "EF":
                            ef_weights[idx][y][name] = level
                        elif sol_target == "Value":
                            val_weights[idx][y][name] = level

                # === MIXED SOLUTIONS ===
                elif sol_type == "mixed":
                    # Handle reduction categories
                    red_sel = set(sol.get("reduction", {}).get("categories", {}).get("checked", []))
                    if is_subpath(label, red_sel):
                        if sol_target == "EF":
                            ef_weights[idx][y][name] = level
                        elif sol_target == "Value":
                            val_weights[idx][y][name] = level

                    # Handle one or multiple increase groups
                    increase_groups = sol.get("increase", [])
                    if isinstance(increase_groups, dict):
                        # Backward compatibility (old format)
                        increase_groups = [increase_groups]

                    for inc_group in increase_groups:
                        inc_sel = set(inc_group.get("categories", {}).get("checked", []))
                        if is_subpath(label, inc_sel):
                            if sol_target == "EF":
                                ef_weights[idx][y][name] = level
                            elif sol_target == "Value":
                                val_weights[idx][y][name] = level

    return ef_weights, val_weights



# =========================================================
# 🔧 Utilities
# =========================================================
def keep_only_most_specific(paths):
    """Keep only the deepest non-redundant hierarchical paths."""
    sorted_paths = sorted(paths, key=lambda x: len(x), reverse=True)
    kept = []
    for p in sorted_paths:
        if not any(p.startswith(k + " >") or p == k for k in kept):
            kept.append(p)
    return kept


def interpolate_targets(year_targets, all_years, start_year, show_debug=False):
    """
    Interpolate target values across all years based on manually defined target points.

    This version includes an optional debug mode to display the interpolation table and graph.

    Parameters:
    - year_targets (dict): {"2026": 0.3, "2028": 0.7}
    - all_years (List[int]): [2025, ..., 2035]
    - start_year (int): first active year
    - show_debug (bool): if True, displays interpolation results with Streamlit

    Returns:
    - dict: {year: proportion (0–1)}
    """
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt

    interpolated = {}
    if not year_targets:
        interpolated = {y: 0.0 for y in all_years}
        if show_debug:
            st.info("⚠️ No year targets defined.")
        return interpolated

    # Convert string keys to int
    year_targets_int = {int(k): v for k, v in year_targets.items()}

    # Ensure interpolation starts at start_year
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

    # 🧭 Optional debug display
    if show_debug:
        st.markdown("#### 🧭 Interpolation debug")
        df = pd.DataFrame.from_dict(interpolated, orient="index", columns=["Effect level (0–1)"])
        st.dataframe(df)

        """fig, ax = plt.subplots()
        ax.plot(df.index, df["Effect level (0–1)"], marker="o")
        ax.set_title("Interpolation curve")
        ax.set_xlabel("Year")
        ax.set_ylabel("Proportion (0–1)")
        st.pyplot(fig)"""

    return interpolated


def is_subpath(path, selected_paths):
    """Check whether 'path' is or is nested under any of 'selected_paths'."""
    return any(path == sel or path.startswith(sel + " >") for sel in selected_paths)


def get_label_path(row):
    """Construct full hierarchical label from a DataFrame row."""
    parts = [
        row.get("Category"),
        row.get("Sub-category 1"),
        row.get("Sub-category 2"),
        row.get("Sub-category 3"),
        row.get("Name"),
        row.get("Location")
    ]
    return " > ".join(str(p).strip() for p in parts if pd.notna(p))


# =========================================================
# 🔢 Emissions & Attribution Calculations
# =========================================================
def compute_emissions_per_year(df, years):
    """Compute annual emissions per row by multiplying EF × Value."""
    emissions_df = df.copy()
    for y in years:
        emissions_df[f"Emissions_{y}"] = df[f"EF_{y}"] * df[f"Value_{y}"]
    return emissions_df


def compute_avoided_emissions(df_before, df_after, years):
    """Compute avoided emissions by comparing before/after values."""
    avoided_df = df_before[[c for c in df_before.columns if "Emissions_" in c]].copy()
    for y in years:
        col = f"Emissions_{y}"
        avoided_df[col] = df_before[col] - df_after[col]
    return avoided_df


def build_diagnostic_weights_table(df, years, ef_weights, val_weights):
    """Build diagnostic DataFrame listing weights per row and year."""
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
    Attribute real avoided emissions to each solution using diagnostic weights.
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

            if brut_total == 0:
                continue

            # EF-based attribution
            total_ef_weight = sum(ef_dict.values())
            for sol, w in ef_dict.items():
                share = w / total_ef_weight if total_ef_weight else 0
                real_impact = share * (brut_ef / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

            # Value-based attribution
            total_val_weight = sum(val_dict.values())
            for sol, w in val_dict.items():
                share = w / total_val_weight if total_val_weight else 0
                real_impact = share * (brut_val / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

    final = pd.DataFrame.from_dict(impact_by_solution, orient="index").fillna(0.0)
    final = final[[y for y in years if y in final.columns]]
    final.index.name = "Solution"
    return final
