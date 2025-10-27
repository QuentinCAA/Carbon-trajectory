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
