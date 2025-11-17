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
# 1️⃣ INITIALIZATION (with stable UUIDs)
# =========================================================
def init_solutions():
    """
    Initialize the default list of mitigation solutions in session state if not already present.
    Each solution now has a stable 'id' (UUID) to ensure widget keys remain stable across
    reordering and JSON save/restore cycles.

    Backward compatibility:
    - If solutions exist but some lack 'id', we assign one on the fly without altering other fields.
    """
    import uuid

    if "solutions" not in st.session_state:
        st.session_state.solutions = [
            {
                "id": str(uuid.uuid4()),
                "name": "Green procurement policy",
                "type": "simple",
                "decarbonation_potential": 0.2,
                "target": "EF",
                "years_targets": {},
                "categories": {}
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Reduced purchasing volumes",
                "type": "simple",
                "decarbonation_potential": 0.25,
                "target": "Value",
                "years_targets": {},
                "categories": {}
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Bike purchase incentive",
                "type": "simple",
                "decarbonation_potential": 0.3,
                "target": "Value",
                "years_targets": {},
                "categories": {}
            }
        ]
    else:
        # Backward compatibility: ensure every solution has a stable id.
        for sol in st.session_state.solutions:
            if "id" not in sol or not sol["id"]:
                sol["id"] = str(uuid.uuid4())
                
# =========================================================
# 2️⃣ CREATION (assign a stable UUID at creation time)
# =========================================================
def create_solution():
    """
    Create a new mitigation solution and store it in session state.
    Uses a stable 'id' (UUID) so reordering and JSON persistence remain robust.

    - Decarbonation potential is stored as a ratio (0–1).
    - Compatible with simple and mixed solutions (new mixed format with increase groups list).
    """
    import uuid

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
                "id": str(uuid.uuid4()),  # 🔒 stable identity
                "name": name,
                "type": solution_type,
                "decarbonation_potential": decarb_potential / 100.0,  # store as ratio
                "target": target,
                "years_targets": {},
                "categories": {},
            }

            if solution_type == "mixed":
                new_solution["reduction"] = {"categories": {}}
                # Use list of increase groups for flexibility
                new_solution["increase"] = []  # each item: {"label", "categories", "conversion_factor"}

            if "solutions" not in st.session_state:
                st.session_state.solutions = []

            st.session_state.solutions.append(new_solution)
            st.success(f"✅ Solution '{name}' ({solution_type}) created successfully.")

# =========================================================
# 3️⃣ CONFIGURATION / ASSIGNMENT (stable keys + clean actions)
# =========================================================

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:52:12 2025

@author: quent
"""

def select_solution(data, years):
    """
    Configure, rename, reorder, or delete existing mitigation solutions.

    Features:
    ----------
    - Each solution includes an editable 'description' field.
    - By default, only key info (number, name, type, target, optional description) is shown.
    - Full details are hidden in an expandable section.
    - Solutions can be reordered or deleted.
    - Stable UUIDs ensure no widget key conflicts.
    """
    import uuid
    import re
    from copy import deepcopy
    from streamlit_tree_select import tree_select

    st.subheader("⚙️ Configure existing solutions")

    st.markdown(
        """
        <div style='color: grey; font-size: 0.9em;'>
        💡 You can rename, describe, reorder, or delete solutions here.<br>
        Click <strong>Expand details</strong> to view and edit the full configuration.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Safety check
    if "solutions" not in st.session_state or not st.session_state.solutions:
        st.info("No solutions available yet. Please create one first.")
        return

    # --- Ensure stable id, description and structure for all solutions
    for sol in st.session_state.solutions:
        if "id" not in sol or not sol["id"]:
            sol["id"] = str(uuid.uuid4())
        if "description" not in sol:
            sol["description"] = ""
        if sol.get("type") == "mixed":
            inc = sol.get("increase", [])
            if isinstance(inc, dict):
                sol["increase"] = [inc]
            elif inc is None:
                sol["increase"] = []

    # Build category tree once
    tree = build_tree(data)

    # Helper to find index by id
    def _idx_of(solutions, sid):
        for ii, ss in enumerate(solutions):
            if ss["id"] == sid:
                return ii
        return None

    cols = st.columns(3)

    # === Render one card per solution ===
    for i, sol in enumerate(st.session_state.solutions):
        sid = sol["id"]
        col = cols[i % 3]
        local = deepcopy(sol)

        with col.container(border=True):
            # ───────────────────────────────────────────────
            # Header: basic info (always visible)
            # ───────────────────────────────────────────────
            st.markdown(f"### 💡 {i + 1}. {local['name']}")
            st.markdown(f"**Type:** {local['type']} | **Target:** {local['target']}")

            # Optional description preview
            if local.get("description"):
                st.markdown(
                    f"<div style='color: grey; font-size: 0.9em;'>{local['description']}</div>",
                    unsafe_allow_html=True,
                )

            # ───────────────────────────────────────────────
            # Buttons row (reorder / delete)
            # ───────────────────────────────────────────────
            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                if st.button("⬆️ Move up", key=f"btn_up_{sid}", use_container_width=True) and i > 0:
                    sols = st.session_state.solutions
                    sols[i - 1], sols[i] = sols[i], sols[i - 1]
                    st.session_state.solutions = sols
                    st.rerun()
            with b2:
                if st.button("⬇️ Move down", key=f"btn_down_{sid}", use_container_width=True) and i < len(st.session_state.solutions) - 1:
                    sols = st.session_state.solutions
                    sols[i + 1], sols[i] = sols[i], sols[i + 1]
                    st.session_state.solutions = sols
                    st.rerun()
            with b3:
                if st.button("🗑️ Delete", key=f"btn_del_{sid}", type="secondary", use_container_width=True):
                    deleted_name = st.session_state.solutions[i]["name"]
                    del st.session_state.solutions[i]
                    st.warning(f"🗑️ Solution '{deleted_name}' deleted.")
                    st.rerun()

            # ───────────────────────────────────────────────
            # Expandable section for detailed configuration
            # ───────────────────────────────────────────────
            with st.expander("🔧 Expand details", expanded=False):
                with st.form(f"form_edit_solution_{sid}"):
                    # --- Editable name & description
                    local["name"] = st.text_input("Solution name", value=local["name"], key=f"name_{sid}")
                    local["description"] = st.text_area(
                        "Description (optional)",
                        value=local.get("description", ""),
                        key=f"desc_{sid}",
                        height=80,
                        help="Briefly describe the purpose or scope of this solution."
                    )

                    st.markdown(f"- **Type:** `{local['type']}` | **Target:** `{local['target']}`")

                    # --- Decarbonation potential
                    decarb_pct = st.number_input(
                        "Decarbonation potential (%) — e.g. 20 = 20% max reduction",
                        min_value=0.0, max_value=100.0,
                        value=float(local.get("decarbonation_potential", 0.0) * 100),
                        format="%.2f",
                        key=f"potential_{sid}"
                    )
                    local["decarbonation_potential"] = decarb_pct / 100.0

                    # --- Start year
                    start_year = local.get("start_year", years[0])
                    start_year = start_year if start_year in years else years[0]
                    local["start_year"] = st.selectbox(
                        "Start year",
                        years,
                        index=years.index(start_year),
                        key=f"start_{sid}"
                    )

                    # --- Implementation targets
                    st.markdown("### Implementation level per year")
                    available_years = [y for y in years if y >= local["start_year"]]
                    year_targets = (local.get("years_targets", {}) or {})
                    year_targets = {str(k): float(v) for k, v in year_targets.items()}

                    selected_years = st.multiselect(
                        "Select target years",
                        available_years,
                        default=sorted(int(y) for y in year_targets.keys()),
                        key=f"years_{sid}"
                    )

                    local_targets = {}
                    for y in selected_years:
                        pct = st.number_input(
                            f"{y} (% of max effect)",
                            min_value=0.0, max_value=100.0,
                            value=float(year_targets.get(str(y), 0.0) * 100),
                            format="%.2f",
                            key=f"{sid}_impl_{y}"
                        )
                        local_targets[str(y)] = pct / 100.0
                    local["years_targets"] = local_targets

                    # --- Category selection
                    if local["type"] == "simple":
                        st.markdown("### Categories impacted by this solution")
                        selection = tree_select(
                            tree,
                            checked=local.get("categories", {}).get("checked", []),
                            expanded=local.get("categories", {}).get("expanded", []),
                            key=f"tree_simple_{sid}"
                        )
                        local["categories"] = selection

                        save_clicked = st.form_submit_button("💾 Save configuration")
                        if save_clicked:
                            idx = _idx_of(st.session_state.solutions, sid)
                            if idx is not None:
                                st.session_state.solutions[idx] = local
                                st.success(f"✅ Configuration for '{local['name']}' saved.")
                                st.rerun()

                    elif local["type"] == "mixed":
                        st.markdown("### 📉 Categories to reduce")
                        reduction = tree_select(
                            tree,
                            checked=local.get("reduction", {}).get("categories", {}).get("checked", []),
                            expanded=local.get("reduction", {}).get("categories", {}).get("expanded", []),
                            key=f"tree_red_{sid}"
                        )
                        local["reduction"] = {"categories": reduction}

                        st.markdown("### 📈 Categories to increase")
                        increase_groups = local.get("increase", [])
                        if isinstance(increase_groups, dict):
                            increase_groups = [increase_groups]
                        if increase_groups is None:
                            increase_groups = []

                        updated_increase_groups = []
                        for j, inc in enumerate(increase_groups):
                            st.markdown(f"#### ➕ Increase group {j+1}")
                            label = st.text_input(
                                "Label (e.g. Boat, Truck, Train)",
                                value=inc.get("label", f"Increase {j+1}"),
                                key=f"inc_label_{sid}_{j}"
                            )
                            factor = st.number_input(
                                "Conversion factor (e.g. 1.5 = 1.5 km replacement per km reduced)",
                                min_value=0.01, format="%.2f",
                                value=float(inc.get("conversion_factor", 1.0)),
                                key=f"factor_{sid}_{j}"
                            )
                            inc_selection = tree_select(
                                tree,
                                checked=inc.get("categories", {}).get("checked", []),
                                expanded=inc.get("categories", {}).get("expanded", []),
                                key=f"tree_inc_{sid}_{j}"
                            )
                            remove = st.checkbox(f"❌ Remove this increase group", key=f"remove_inc_{sid}_{j}")
                            if not remove:
                                updated_increase_groups.append({
                                    "label": label,
                                    "categories": inc_selection,
                                    "conversion_factor": factor
                                })

                        add_clicked = st.form_submit_button("➕ Add new increase group")
                        save_clicked = st.form_submit_button("💾 Save configuration")

                        if add_clicked:
                            idx = _idx_of(st.session_state.solutions, sid)
                            if idx is not None:
                                current = st.session_state.solutions[idx]
                                cur_inc = current.get("increase", [])
                                if isinstance(cur_inc, dict):
                                    cur_inc = [cur_inc]
                                cur_inc.append({
                                    "label": f"Increase {len(cur_inc)+1}",
                                    "categories": {"checked": [], "expanded": []},
                                    "conversion_factor": 1.0
                                })
                                current["increase"] = cur_inc
                                st.session_state.solutions[idx] = current
                                st.rerun()

                        if save_clicked:
                            local["increase"] = updated_increase_groups
                            idx = _idx_of(st.session_state.solutions, sid)
                            if idx is not None:
                                st.session_state.solutions[idx] = local
                                st.success(f"✅ Configuration for '{local['name']}' saved.")
                                st.rerun()



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
                        per_row_increase = total_increase / len(affected_rows)
                        for idx in affected_rows:
                            modified_df.at[idx, col] += per_row_increase

        st.markdown("---")

    return modified_df


def apply_single_solution(df, years, sol):
    """
    Apply a *single* mitigation solution to a DataFrame in place.

    This helper mirrors the logic used in `apply_solutions`, but only for one
    solution at a time and WITHOUT any Streamlit UI.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of projected data BEFORE applying this solution.
        It is modified in place.
    years : list[int]
        List of years used to build column names like 'EF_2025', 'Value_2027', etc.
    sol : dict
        Solution configuration, as stored in `st.session_state.solutions`.
        Must contain at least:
        - 'name'
        - 'type'  ('simple' or 'mixed')
        - 'target' ('EF' or 'Value')
        - 'decarbonation_potential' (ratio 0–1)
        - 'years_targets' (dict) and optionally 'start_year'
        - category selections in 'categories' (simple) or
          'reduction' / 'increase' (mixed).
    """
    # Import here to avoid circular issues if this module is reused
    import pandas as pd

    name = sol.get("name", "Unnamed")
    potential = sol.get("decarbonation_potential", 0.0)
    target_field = sol.get("target", "EF")

    if potential == 0:
        # No effect at all if potential is zero
        return

    # --- SIMPLE SOLUTION -----------------------------------------------------
    if sol.get("type") == "simple":
        raw_targets = sol.get("years_targets", {}) or {}
        start_year = sol.get("start_year", years[0])

        # Interpolate implementation level for all years
        interpolated_targets = interpolate_targets(
            raw_targets, years, start_year, show_debug=False
        )

        # Selected categories for this solution
        selected = set(sol.get("categories", {}).get("checked", []))

        for idx, row in df.iterrows():
            full_label = get_label_path(row)
            if not is_subpath(full_label, selected):
                continue

            for year in years:
                col = f"{target_field}_{year}"
                if col not in df.columns:
                    continue

                # Final reduction factor for this year = potential × implementation level
                reduction = potential * interpolated_targets.get(year, 0.0)
                if reduction <= 0:
                    continue

                before_value = df.at[idx, col]
                # If a cell is NaN or zero, skip to avoid propagating noise
                if pd.isna(before_value) or before_value == 0:
                    continue

                df.at[idx, col] = before_value * (1 - reduction)

    # --- MIXED SOLUTION ------------------------------------------------------
    elif sol.get("type") == "mixed":
        raw_targets = sol.get("years_targets", {}) or {}
        start_year = sol.get("start_year", years[0])

        interpolated_targets = interpolate_targets(
            raw_targets, years, start_year, show_debug=False
        )

        # Reduction side
        reduction_paths = set(
            sol.get("reduction", {}).get("categories", {}).get("checked", [])
        )

        # Increase side: list of groups ({label, categories, conversion_factor})
        increase_groups = sol.get("increase", [])
        if isinstance(increase_groups, dict):
            # Backward compatibility if 'increase' was stored as a dict
            increase_groups = [increase_groups]
        if increase_groups is None:
            increase_groups = []

        # We will first apply reductions and accumulate "capacity" by year,
        # then redistribute this capacity across increase groups.
        yearly_reductions = {y: 0.0 for y in years}

        # --- Phase 1: apply reductions on selected rows ----------------------
        for idx, row in df.iterrows():
            full_label = get_label_path(row)
            if not is_subpath(full_label, reduction_paths):
                continue

            for year in years:
                col = f"{target_field}_{year}"
                if col not in df.columns:
                    continue

                before_value = df.at[idx, col]
                if pd.isna(before_value) or before_value == 0:
                    continue

                reduction = potential * interpolated_targets.get(year, 0.0)
                if reduction <= 0:
                    continue

                delta = before_value * reduction
                df.at[idx, col] = before_value - delta
                yearly_reductions[year] += delta

        # --- Phase 2: redistribute increased activity across increase groups --
        # For each group, we allocate a share of the yearly_reductions scaled
        # by the group conversion factor, and distribute it evenly on all
        # rows that belong to the group.
        for inc_group in increase_groups:
            factor = inc_group.get("conversion_factor", 1.0) or 1.0
            increase_paths = set(
                inc_group.get("categories", {}).get("checked", [])
            )

            # Identify all rows that belong to this increase group
            affected_rows = [
                idx for idx, row in df.iterrows()
                if is_subpath(get_label_path(row), increase_paths)
            ]
            if not affected_rows:
                continue

            for year in years:
                col = f"{target_field}_{year}"
                if col not in df.columns:
                    continue

                total_increase = yearly_reductions[year] * factor
                if total_increase == 0:
                    continue

                per_row_increase = total_increase / len(affected_rows)
                for idx in affected_rows:
                    before_value = df.at[idx, col]
                    if pd.isna(before_value):
                        before_value = 0.0
                    df.at[idx, col] = before_value + per_row_increase

    # If sol["type"] is something else, we silently ignore it for now


def build_solution_weights_table(df, years, st_session_solutions):
    """
    Build tables of *isolated* solution impacts per row and year.

    New logic (isolated impact approach)
    ------------------------------------
    For each solution, we do:

        1. Start from the baseline DataFrame `df` (before any solution).
        2. Make a copy of `df` -> `df_iso`.
        3. Apply ONLY THIS solution to `df_iso` (using `apply_single_solution`).
        4. For each row and year, compute the change in EF and in Value
           between baseline and this isolated scenario.
        5. Convert those changes into emissions impact:

            - If solution target is "EF":
                  impact = (EF_before - EF_after) * Value_before

            - If solution target is "Value":
                  impact = (Value_before - Value_after) * EF_before

        6. Store this impact as the weight for that row/year/solution.

    IMPORTANT
    ---------
    - We now keep **all impacts**, including negative ones:
        - Positive  => avoided emissions.
        - Negative  => additional emissions caused by the solution
                       (e.g. when shifting activity to another line).
    - This avoids artificially inflating / deflating the importance of a
      mixed solution by ignoring parts where it increases emissions.

    Output structure
    ----------------
    We keep the same structure for compatibility with the rest of the code:

        ef_weights[row_index][year][solution_name]  = isolated impact
                                                     due to EF changes only
        val_weights[row_index][year][solution_name] = isolated impact
                                                     due to Value changes only

    Parameters
    ----------
    df : pd.DataFrame
        Baseline projection BEFORE applying any solution.
    years : list[int]
        List of years used for EF/Value/Emissions columns.
    st_session_solutions : list[dict]
        List of solution configurations (e.g. `st.session_state.solutions`).

    Returns
    -------
    (ef_weights, val_weights) : (dict, dict)
        Nested dictionaries as described above.
    """
    import pandas as pd

    # Initialize nested dicts for all rows and years
    ef_weights = {idx: {y: {} for y in years} for idx in df.index}
    val_weights = {idx: {y: {} for y in years} for idx in df.index}

    # Precompute baseline EF and Value for speed and clarity
    baseline_ef = {
        y: df[f"EF_{y}"].copy() for y in years if f"EF_{y}" in df.columns
    }
    baseline_val = {
        y: df[f"Value_{y}"].copy() for y in years if f"Value_{y}" in df.columns
    }

    for sol in st_session_solutions:
        name = sol.get("name", "Unnamed solution")
        sol_target = sol.get("target", "EF")

        # Skip solutions with zero potential
        potential = sol.get("decarbonation_potential", 0.0)
        if potential == 0:
            continue

        # Work on a copy of the baseline
        df_iso = df.copy()

        # Apply ONLY this solution on df_iso
        apply_single_solution(df_iso, years, sol)

        # For each row/year, compute isolated impact
        for idx in df.index:
            for y in years:
                ef_col = f"EF_{y}"
                val_col = f"Value_{y}"

                if ef_col not in df_iso.columns or val_col not in df_iso.columns:
                    continue

                # Baseline (before)
                ef_before = baseline_ef[y].get(idx, None)
                val_before = baseline_val[y].get(idx, None)

                if pd.isna(ef_before) or pd.isna(val_before):
                    continue

                # After applying only this solution
                ef_after = df_iso.at[idx, ef_col]
                val_after = df_iso.at[idx, val_col]

                # For EF-targeted solutions, we attribute impact through EF changes:
                #   impact = (EF_before - EF_after) * Value_before
                if sol_target == "EF":
                    impact_ef = (ef_before - ef_after) * val_before

                    # ✅ Keep impacts even if they are negative or zero:
                    #    positive  -> avoided emissions
                    #    negative  -> additional emissions
                    if impact_ef != 0:
                        ef_weights[idx][y][name] = (
                            ef_weights[idx][y].get(name, 0.0) + impact_ef
                        )

                # For Value-targeted solutions, we attribute impact through Value changes:
                #   impact = (Value_before - Value_after) * EF_before
                elif sol_target == "Value":
                    impact_val = (val_before - val_after) * ef_before

                    if impact_val != 0:
                        val_weights[idx][y][name] = (
                            val_weights[idx][y].get(name, 0.0) + impact_val
                        )

                # If sol_target is something else, we ignore it for now.

    return ef_weights, val_weights

def build_weights_debug_table(ef_weights, val_weights, years):
    """
    Build a long-format debug table of non-zero solution weights.

    This function flattens the nested dictionaries produced by
    `build_solution_weights_table` into a tabular format:

        - one row per (row_index, year, field, solution)
        - only non-zero weights are kept
        - weights can be positive (avoided emissions) or negative
          (additional emissions).

    Parameters
    ----------
    ef_weights : dict
        Nested dictionary:
            ef_weights[row_index][year][solution_name] = weight
        Typically represents impacts due to changes in EF.
    val_weights : dict
        Nested dictionary:
            val_weights[row_index][year][solution_name] = weight
        Typically represents impacts due to changes in Value.
    years : list[int]
        List of years to include in the debug table.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - 'Row index'
            - 'Year'
            - 'Field'   ('EF' or 'Value')
            - 'Solution'
            - 'Weight'
        Only rows with Weight != 0 are included.
    """
    import pandas as pd

    debug_rows = []

    # --- Flatten EF weights --------------------------------------------------
    for idx, year_dict in ef_weights.items():
        for y in years:
            # Safely access the dictionary for this year
            sol_dict = year_dict.get(y, {})
            if not sol_dict:
                continue

            for sol_name, w in sol_dict.items():
                if w == 0 or w is None:
                    continue

                debug_rows.append({
                    "Row index": idx,
                    "Year": y,
                    "Field": "EF",
                    "Solution": sol_name,
                    "Weight": w,
                })

    # --- Flatten Value weights ----------------------------------------------
    for idx, year_dict in val_weights.items():
        for y in years:
            sol_dict = year_dict.get(y, {})
            if not sol_dict:
                continue

            for sol_name, w in sol_dict.items():
                if w == 0 or w is None:
                    continue

                debug_rows.append({
                    "Row index": idx,
                    "Year": y,
                    "Field": "Value",
                    "Solution": sol_name,
                    "Weight": w,
                })

    # Build DataFrame
    if not debug_rows:
        return pd.DataFrame(
            columns=["Row index", "Year", "Field", "Solution", "Weight"]
        )

    df_debug = pd.DataFrame(debug_rows)

    # Sort for readability: by solution, then year, then field, then row index
    df_debug = df_debug.sort_values(
        by=["Solution", "Year", "Field", "Row index"]
    ).reset_index(drop=True)

    return df_debug

def build_weights_summary_table(debug_df):
    """
    Build an aggregated summary table from the debug weights table.

    Parameters
    ----------
    debug_df : pd.DataFrame
        Output of `build_weights_debug_table`, with columns:
        ['Row index', 'Year', 'Field', 'Solution', 'Weight'].

    Returns
    -------
    pd.DataFrame
        Aggregated table with:
            - Solution
            - Year
            - Field ('EF' or 'Value')
            - Total weight (sum of weights for this group)
    """
    if debug_df.empty:
        return debug_df

    summary = (
        debug_df
        .groupby(["Solution", "Year", "Field"], as_index=False)["Weight"]
        .sum()
        .rename(columns={"Weight": "Total weight"})
        .sort_values(by=["Solution", "Year", "Field"])
        .reset_index(drop=True)
    )

    return summary


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

        fig, ax = plt.subplots()
        ax.plot(df.index, df["Effect level (0–1)"], marker="o")
        ax.set_title("Interpolation curve")
        ax.set_xlabel("Year")
        ax.set_ylabel("Proportion (0–1)")
        st.pyplot(fig)

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

    This version stabilises the EF / Value shares when the sum of weights is
    very small (asymptote problem: positive and negative weights almost cancel).

    Heuristic:
        - We compute share_s = w_s / sum(w).
        - If any |share_s| > MAX_ABS_SHARE (default 3),
          we *double* the smallest |w_s| and recompute.
        - This pushes the denominator away from 0 without changing the
          distribution too much (we only touch the smallest weight).

    If at least one stabilisation is applied, a warning message is displayed
    in the Streamlit UI to inform the user that a numerical adjustment has
    been used (no exact mathematical solution for this case).
    """
    import math
    import pandas as pd

    MAX_ABS_SHARE = 3.0
    MAX_ITER = 50

    # This list will collect contexts where stabilisation was actually applied
    stabilisation_log = []

    def stabilise_shares(weights_dict, context_label, max_abs_share=MAX_ABS_SHARE, max_iter=MAX_ITER):
        """
        Given a dict {solution: weight}, return a dict {solution: share}
        such that:
            - share_s ≈ weight_s / sum(weights)
            - but no |share_s| is larger than max_abs_share
              (unless we hit max_iter or a degenerate case).

        If we cannot normalise (all zero / sum == 0), returns zeros.

        If stabilisation is needed (at least one modification of weights),
        the context_label is appended to the outer stabilisation_log.
        """
        # Work on a copy so we do not mutate the original dictionary
        weights = dict(weights_dict)

        # Edge case: empty dict
        if not weights:
            return {}

        modified = False  # track whether we actually changed the weights

        for _ in range(max_iter):
            total = sum(weights.values())
            if math.isclose(total, 0.0, abs_tol=1e-15):
                # Degenerate case: cannot normalise, return all zeros
                return {s: 0.0 for s in weights}

            shares = {s: w / total for s, w in weights.items()}
            max_share = max(abs(sh) for sh in shares.values())

            # If all shares are within the acceptable range, we are done
            if max_share <= max_abs_share:
                # If we had to modify weights to reach this state,
                # log the context for user information later.
                if modified and context_label is not None:
                    stabilisation_log.append(context_label)
                return shares

            # Otherwise, we push the denominator away from zero by
            # doubling the smallest |w| (in absolute value).
            # This should change the distribution minimally.
            non_zero_weights = [(s, w) for s, w in weights.items() if w != 0]
            if not non_zero_weights:
                # All weights are zero -> cannot fix, return zeros
                return {s: 0.0 for s in weights}

            # Solution with the smallest absolute weight
            sol_min, w_min = min(non_zero_weights, key=lambda item: abs(item[1]))
            weights[sol_min] = w_min * 2
            modified = True

        # If we exit the loop without satisfying the condition, just
        # return the last computed shares and log the context if modified.
        total = sum(weights.values())
        if math.isclose(total, 0.0, abs_tol=1e-15):
            return {s: 0.0 for s in weights}

        if modified and context_label is not None:
            stabilisation_log.append(context_label)

        return {s: w / total for s, w in weights.items()}

    impact_by_solution = {}

    for idx in df_before.index:
        for year in years:
            ef_col = f"EF_{year}"
            val_col = f"Value_{year}"
            em_col = f"Emissions_{year}"

            # Retrieve before/after values
            ef_b = df_before.at[idx, ef_col]
            ef_a = df_after.at[idx, ef_col]
            val_b = df_before.at[idx, val_col]
            val_a = df_after.at[idx, val_col]

            delta = df_avoided.at[idx, em_col]
            if delta == 0:
                # Nothing to attribute for this row/year
                continue

            # Retrieve diagnostic weights (lists of (solution, percentage))
            key_ef = f"{idx} - EF"
            key_val = f"{idx} - Value"
            ef_weights = diagnostic_df.loc[key_ef, year] if key_ef in diagnostic_df.index else []
            val_weights = diagnostic_df.loc[key_val, year] if key_val in diagnostic_df.index else []

            # Convert to dictionaries of raw weights in [0,1]
            ef_dict = {s: pct / 100 for s, pct in ef_weights} if isinstance(ef_weights, list) else {}
            val_dict = {s: pct / 100 for s, pct in val_weights} if isinstance(val_weights, list) else {}

            # Compute brut effects
            brut_ef = (ef_b - ef_a) * val_b
            brut_val = (val_b - val_a) * ef_b
            brut_total = brut_ef + brut_val

            if math.isclose(brut_total, 0.0, abs_tol=1e-15):
                # EF and Value effects cancel each other -> no stable attribution
                continue

            # -------------------------
            # EF-based attribution
            # -------------------------
            ef_shares = stabilise_shares(
                ef_dict,
                context_label=f"Row {idx}, year {year}, field EF",
            )
            for sol, share in ef_shares.items():
                # share already includes the normalisation; we just apply
                # the fraction of delta that comes from EF
                real_impact = share * (brut_ef / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

            # -------------------------
            # Value-based attribution
            # -------------------------
            val_shares = stabilise_shares(
                val_dict,
                context_label=f"Row {idx}, year {year}, field Value",
            )
            for sol, share in val_shares.items():
                real_impact = share * (brut_val / brut_total * delta)
                impact_by_solution.setdefault(sol, {}).setdefault(year, 0.0)
                impact_by_solution[sol][year] += real_impact

    # -------------------------------------------
    # If stabilisation has been used, warn user
    # -------------------------------------------
    if stabilisation_log:
        try:
            import streamlit as st

            st.warning(
                "⚠️ Numerical stabilisation was applied for some rows during the "
                "attribution of avoided emissions to solutions."
            )
            st.caption(
                "Because some diagnostic weights were very close to an asymptotic case "
                "(positive and negative weights almost cancelling out), the attribution "
                "to individual solutions does not have a unique, exact mathematical "
                "solution. A small numerical adjustment was applied to the smallest "
                "weights to move the system away from this asymptote and obtain a "
                "stable allocation."
            )
            # Optional: show how many cases were adjusted (without flooding details)
            st.caption(
                f"Stabilisation was triggered in {len(stabilisation_log)} row/year/field "
                "cases. You can inspect critical rows with the debug tools in the Results tab."
            )
        except Exception:
            # If Streamlit is not available (e.g. running in a pure Python context),
            # just ignore the UI warning.
            pass

    # Build final DataFrame
    final = pd.DataFrame.from_dict(impact_by_solution, orient="index").fillna(0.0)
    final = final[[y for y in years if y in final.columns]]
    final.index.name = "Solution"
    return final



def build_compute_debug_table(df_before, df_after, df_avoided, diagnostic_df, years, row_index):
    """
    Build a debug table showing how avoided emissions are split between EF and Value
    for a single row across all years.

    This function reproduces the internal calculations of
    `compute_solution_impact_from_diagnostic` but only for one row, and
    structures the intermediate values in a readable table.

    One row in the output corresponds to:
        (row_index, year, field) where field ∈ {'EF', 'Value'}

    Columns include:
        - Row index
        - Year
        - Field ('EF' or 'Value')
        - EF_before / EF_after
        - Value_before / Value_after
        - brut_component  (brut_ef or brut_val)
        - brut_total      (brut_ef + brut_val)
        - delta_avoided   (df_avoided Emissions_before - after)
        - Weights_dict    (diagnostic weights for that field & year)
        - Weights_sum
        - Component_ratio (brut_component / brut_total)
        - Flag_small_total (True if brut_total is close to 0)

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame of emissions *before* solutions (EF_x, Value_x, Emissions_x).
    df_after : pd.DataFrame
        DataFrame of emissions *after* solutions.
    df_avoided : pd.DataFrame
        DataFrame of avoided emissions per line (Emissions_before - Emissions_after).
    diagnostic_df : pd.DataFrame
        Diagnostic table where each cell is typically a list of (solution, pct) tuples.
    years : list[int]
        List of years to inspect.
    row_index : hashable
        The index of the row to debug (must be present in df_before / df_after / df_avoided).

    Returns
    -------
    pd.DataFrame
        Long-format debug table for the selected row.
    """
    import math
    import pandas as pd

    debug_rows = []

    if row_index not in df_before.index:
        # Return empty table with correct structure if index is invalid
        return pd.DataFrame(
            columns=[
                "Row index", "Year", "Field",
                "EF_before", "EF_after",
                "Value_before", "Value_after",
                "brut_component", "brut_total",
                "delta_avoided",
                "Weights_dict", "Weights_sum",
                "Component_ratio", "Flag_small_total",
            ]
        )

    for year in years:
        ef_col = f"EF_{year}"
        val_col = f"Value_{year}"
        em_col = f"Emissions_{year}"

        # Skip if some columns are missing
        if ef_col not in df_before.columns or val_col not in df_before.columns:
            continue
        if em_col not in df_avoided.columns:
            continue

        ef_b = df_before.at[row_index, ef_col]
        ef_a = df_after.at[row_index, ef_col]
        val_b = df_before.at[row_index, val_col]
        val_a = df_after.at[row_index, val_col]

        delta = df_avoided.at[row_index, em_col]

        # Skip if absolutely nothing happened for this year
        if (ef_b == ef_a) and (val_b == val_a) and (delta == 0):
            continue

        # --- Retrieve diagnostic weights for this row/year -------------------
        key_ef = f"{row_index} - EF"
        key_val = f"{row_index} - Value"

        # Defensive access to diagnostic_df
        if (key_ef in diagnostic_df.index) and (year in diagnostic_df.columns):
            ef_weights_cell = diagnostic_df.loc[key_ef, year]
        else:
            ef_weights_cell = []

        if (key_val in diagnostic_df.index) and (year in diagnostic_df.columns):
            val_weights_cell = diagnostic_df.loc[key_val, year]
        else:
            val_weights_cell = []

        # Convert (solution, pct) list to dict of weights in [0,1]
        ef_dict = (
            {s: pct / 100 for s, pct in ef_weights_cell}
            if isinstance(ef_weights_cell, list) else {}
        )
        val_dict = (
            {s: pct / 100 for s, pct in val_weights_cell}
            if isinstance(val_weights_cell, list) else {}
        )

        # --- Compute brut effects -------------------------------------------
        brut_ef = (ef_b - ef_a) * val_b
        brut_val = (val_b - val_a) * ef_b
        brut_total = brut_ef + brut_val

        # Check if total is almost zero (cancelling positive & negative effects)
        flag_small_total = math.isclose(brut_total, 0.0, abs_tol=1e-12)

        # EF component row
        component_ratio_ef = None
        if not flag_small_total:
            component_ratio_ef = brut_ef / brut_total if brut_total != 0 else None

        debug_rows.append({
            "Row index": row_index,
            "Year": year,
            "Field": "EF",
            "EF_before": ef_b,
            "EF_after": ef_a,
            "Value_before": val_b,
            "Value_after": val_a,
            "brut_component": brut_ef,
            "brut_total": brut_total,
            "delta_avoided": delta,
            "Weights_dict": ef_dict,
            "Weights_sum": sum(ef_dict.values()),
            "Component_ratio": component_ratio_ef,
            "Flag_small_total": flag_small_total,
        })

        # Value component row
        component_ratio_val = None
        if not flag_small_total:
            component_ratio_val = brut_val / brut_total if brut_total != 0 else None

        debug_rows.append({
            "Row index": row_index,
            "Year": year,
            "Field": "Value",
            "EF_before": ef_b,
            "EF_after": ef_a,
            "Value_before": val_b,
            "Value_after": val_a,
            "brut_component": brut_val,
            "brut_total": brut_total,
            "delta_avoided": delta,
            "Weights_dict": val_dict,
            "Weights_sum": sum(val_dict.values()),
            "Component_ratio": component_ratio_val,
            "Flag_small_total": flag_small_total,
        })

    if not debug_rows:
        return pd.DataFrame(
            columns=[
                "Row index", "Year", "Field",
                "EF_before", "EF_after",
                "Value_before", "Value_after",
                "brut_component", "brut_total",
                "delta_avoided",
                "Weights_dict", "Weights_sum",
                "Component_ratio", "Flag_small_total",
            ]
        )

    df_debug = pd.DataFrame(debug_rows)

    # Sort for readability
    df_debug = df_debug.sort_values(
        by=["Year", "Field"]
    ).reset_index(drop=True)

    return df_debug

def build_row_year_solution_debug(df_before, df_after, df_avoided, diagnostic_df, row_index, year):
    """
    Build a per-solution attribution breakdown for a single row and a single year.

    This function mirrors the logic of `compute_solution_impact_from_diagnostic` but
    only for one (row_index, year) pair. It splits the real avoided emissions (delta)
    into:
        - a part due to EF change,
        - a part due to Value change,
    and then allocates each part between solutions using the diagnostic weights.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame of emissions *before* solutions (EF_year, Value_year, Emissions_year).
    df_after : pd.DataFrame
        DataFrame of emissions *after* solutions.
    df_avoided : pd.DataFrame
        DataFrame of avoided emissions (Emissions_before - Emissions_after).
    diagnostic_df : pd.DataFrame
        Diagnostic table where each cell is typically a list of (solution_name, percentage).
    row_index : hashable
        Index of the row to inspect (must exist in df_before / df_after / df_avoided).
    year : int
        Year to inspect.

    Returns
    -------
    tuple (df_solutions, meta)
        df_solutions : pd.DataFrame
            One row per solution appearing in EF and/or Value weights for this row/year.
            Columns:
                - Solution
                - EF_weight_raw        (sum of raw EF weights in [0,1])
                - EF_weight_norm       (normalised EF weight, sum over solutions = 1 if any EF weights)
                - Value_weight_raw     (sum of raw Value weights in [0,1])
                - Value_weight_norm    (normalised Value weight, sum over solutions = 1 if any Value weights)
                - Impact_from_EF       (tonnes attributed via EF part)
                - Impact_from_Value    (tonnes attributed via Value part)
                - Total_impact         (Impact_from_EF + Impact_from_Value)

        meta : dict
            Dictionary with scalar information for this row/year:
                - 'EF_before', 'EF_after'
                - 'Value_before', 'Value_after'
                - 'delta_avoided'
                - 'brut_ef', 'brut_val', 'brut_total'
                - 'flag_small_total' (True if brut_total ~ 0)
    """
    import math
    import pandas as pd

    ef_col = f"EF_{year}"
    val_col = f"Value_{year}"
    em_col = f"Emissions_{year}"

    # Basic safety checks
    if row_index not in df_before.index:
        return pd.DataFrame(columns=[
            "Solution",
            "EF_weight_raw", "EF_weight_norm",
            "Value_weight_raw", "Value_weight_norm",
            "Impact_from_EF", "Impact_from_Value", "Total_impact",
        ]), {}

    if (ef_col not in df_before.columns or
        val_col not in df_before.columns or
        em_col not in df_avoided.columns):
        return pd.DataFrame(columns=[
            "Solution",
            "EF_weight_raw", "EF_weight_norm",
            "Value_weight_raw", "Value_weight_norm",
            "Impact_from_EF", "Impact_from_Value", "Total_impact",
        ]), {}

    # --- Retrieve before / after values --------------------------------------
    ef_b = df_before.at[row_index, ef_col]
    ef_a = df_after.at[row_index, ef_col]
    val_b = df_before.at[row_index, val_col]
    val_a = df_after.at[row_index, val_col]
    delta = df_avoided.at[row_index, em_col]

    # --- Retrieve diagnostic weights for this row/year -----------------------
    key_ef = f"{row_index} - EF"
    key_val = f"{row_index} - Value"

    if (key_ef in diagnostic_df.index) and (year in diagnostic_df.columns):
        ef_weights_cell = diagnostic_df.loc[key_ef, year]
    else:
        ef_weights_cell = []

    if (key_val in diagnostic_df.index) and (year in diagnostic_df.columns):
        val_weights_cell = diagnostic_df.loc[key_val, year]
    else:
        val_weights_cell = []

    # Convert list of (solution, pct) into dict of raw weights in [0,1]
    ef_dict = (
        {s: pct / 100 for s, pct in ef_weights_cell}
        if isinstance(ef_weights_cell, list) else {}
    )
    val_dict = (
        {s: pct / 100 for s, pct in val_weights_cell}
        if isinstance(val_weights_cell, list) else {}
    )

    # --- Compute brut effects (same as in the main attribution function) -----
    brut_ef = (ef_b - ef_a) * val_b
    brut_val = (val_b - val_a) * ef_b
    brut_total = brut_ef + brut_val

    flag_small_total = math.isclose(brut_total, 0.0, abs_tol=1e-12)

    # Ratios: share of total effect coming from EF vs Value
    if flag_small_total or brut_total == 0:
        ratio_ef = 0.0
        ratio_val = 0.0
    else:
        ratio_ef = brut_ef / brut_total
        ratio_val = brut_val / brut_total

    # --- Normalise weights for EF and Value ----------------------------------
    sum_ef = sum(ef_dict.values())
    sum_val = sum(val_dict.values())

    ef_norm = {s: (w / sum_ef) if sum_ef else 0.0 for s, w in ef_dict.items()}
    val_norm = {s: (w / sum_val) if sum_val else 0.0 for s, w in val_dict.items()}

    # Union of all solutions appearing in EF and/or Value weights
    all_solutions = set(ef_dict.keys()) | set(val_dict.keys())

    rows = []
    for sol in sorted(all_solutions):
        w_ef_raw = ef_dict.get(sol, 0.0)
        w_val_raw = val_dict.get(sol, 0.0)
        w_ef_norm = ef_norm.get(sol, 0.0)
        w_val_norm = val_norm.get(sol, 0.0)

        # Effective impact attributed to this solution
        impact_from_ef = delta * ratio_ef * w_ef_norm
        impact_from_val = delta * ratio_val * w_val_norm
        total_impact = impact_from_ef + impact_from_val

        rows.append({
            "Solution": sol,
            "EF_weight_raw": w_ef_raw,
            "EF_weight_norm": w_ef_norm,
            "Value_weight_raw": w_val_raw,
            "Value_weight_norm": w_val_norm,
            "Impact_from_EF": impact_from_ef,
            "Impact_from_Value": impact_from_val,
            "Total_impact": total_impact,
        })

    df_solutions = pd.DataFrame(rows).sort_values(
        by="Total_impact", ascending=False
    ).reset_index(drop=True)

    meta = {
        "EF_before": ef_b,
        "EF_after": ef_a,
        "Value_before": val_b,
        "Value_after": val_a,
        "delta_avoided": delta,
        "brut_ef": brut_ef,
        "brut_val": brut_val,
        "brut_total": brut_total,
        "flag_small_total": flag_small_total,
        "ratio_ef": ratio_ef,
        "ratio_val": ratio_val,
        "sum_ef_weights": sum_ef,
        "sum_val_weights": sum_val,
    }

    return df_solutions, meta
