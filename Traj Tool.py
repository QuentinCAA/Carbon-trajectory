# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:47:39 2025

@author: quent

This is the main Streamlit app for carbon trajectory projections:
- Import baseline data
- Create and assign growth/budget projections (+ inflation)
- Create and assign structural effects
- Create and configure mitigation solutions (simple & mixed)
- Apply everything to produce after-scenarios
- Attribute avoided emissions to solutions
- Visualize and export results

IMPORTANT ARCHITECTURE NOTE
---------------------------
We now keep a SINGLE source of truth for solution yearly targets:
    st.session_state["solutions"][i]["years_targets"]

We REMOVED any mirrored structures like `solutions_table` or `targets_table`
to avoid drift between what is displayed and what is exported/restored.

See Tab 4 for a small one-time migration function that merges any legacy
mirrors into the solution objects and deletes those mirrors afterwards.
"""

# ================================
# Table of Contents
# ================================
# Development Notes               -> just after the table of contents
# 1. Code Initialization          ->
# 2. Tab 1: Home                  ->
# 3. Tab 2: Growth                ->
# 4. Tab 3: Structural Effects    ->
# 5. Tab 4: Solutions             ->
# 6. Tab 5: Results               ->
# 7. Tab 6: Visualisations        ->
# 8. Tab 7: Export                ->
# ================================


# =========================================
# Development Notes (update every time you close the file)
# =========================================

## To Do
#- [ ] Display nice charts
#- [ ] Give the possibility to export data for financial trajectory (tableau excel avec growths et solutions avec ttes les infos disponibles pour les deux)
#- [ ] Create an utilisation guide (with screenshots) and by doing so also facilitate UX with more explanations at the start of each tab
#- [ ] Create example JSONs for training/understanding in the previous guide
#- [ ] Clarify databefore and dataafter to calculate the reduction with growth and structural effects different roles (see with Paolo)
#- [ ] Create a link between inflation and localisation ?

## Done
#- [X] Enable input of growth forecasts (with multiple possible growth scenarios); determine how to assign growth to categories/subcategories
#- [X] Handle import of new input format and allow simple visualisation by category
#- [X] Enable input of structural effects and manage their assignment
#- [X] Allow users to create solutions (simple and mix)
#- [X] Calculate the emissions after the solutions attribution
#- [X] Display the impact of each solution
#- [X] Display projected values by name and year
#- [X] Allow export of a file to avoid starting from scratch
#- [X] Review code and comment properly
#- [X] Integrate inflation
#- [X] Remove duplicate targets tables and keep a single source of truth


# =========================================
# 1. Code Initialization
# =========================================

import streamlit as st
import pandas as pd
import json
from io import BytesIO

from modules.colors import (
    choose_colors, show_pie_chart_by_category, show_total_emissions
)
from modules.tree import create_projection_base
from modules.growth import (
    create_growth, assign_growth, apply_projections_to_base,
    check_projection_coverage, define_inflation, summarize_growths
)
from modules.structural import (
    init_structural_effects, create_structural_effect, assign_structural_effects,
    apply_structural_effects, check_structural_coverage, compute_structural_impact
)
from modules.solutions import (
    init_solutions, select_solution, apply_solutions, create_solution,
    compute_avoided_emissions, compute_emissions_per_year,
    build_diagnostic_weights_table, build_solution_weights_table,
    compute_solution_impact_from_diagnostic, build_weights_summary_table , build_weights_debug_table
)
from modules.visualisation import (
    choose_solution_colors_and_order, plot_cumulative_emissions_reduction,
    plot_annual_emissions_reduction, prepare_waterfall_inputs,
    plot_waterfall_emissions, export_svg, compute_solution_percentages
)

# Activate wide layout mode to reduce side margins (must be the first Streamlit command)
st.set_page_config(layout="wide")

# Main tabs
tabs = st.tabs(["Home", "Growth", "Structural Effects", "Solutions", "Results", "Visualisations", "Export"])

# Helper: check if session is ready
def has_loaded_data():
    """True if we have both a baseline dataset and a list of projection years."""
    return "data" in st.session_state and "years" in st.session_state

def run_results_computation():
    """
    Run the heavy results pipeline once and cache all intermediate outputs
    in Streamlit session state.

    This function:
    - Computes emissions before and after solutions.
    - Computes avoided emissions.
    - Builds EF/Value weight tables per solution (isolated impact method).
    - Builds the diagnostic matrix used for attribution.
    - Computes the final impact per solution and per year.

    It MUST be called explicitly (via a button) to avoid recalculating
    everything on every small UI change (especially in the Solutions tab).
    """
    if not has_loaded_data():
        st.warning("No data loaded. Please upload your footprint file first.")
        return

    years = st.session_state["years"]
    projected_with_structural = st.session_state.get("projected_with_structural")
    projected_with_solutions = st.session_state.get("projected_with_solutions")

    if projected_with_structural is None or projected_with_solutions is None:
        st.warning(
            "Missing intermediate scenarios. Please make sure you have completed "
            "Growth and Structural Effects tabs before running the results."
        )
        return

    # 1) Emissions before / after solutions
    df_emissions_before = compute_emissions_per_year(projected_with_structural, years)
    df_emissions_after = compute_emissions_per_year(projected_with_solutions, years)
    df_avoided = compute_avoided_emissions(df_emissions_before, df_emissions_after, years)

    # 2) Build solution weight tables (isolated impact approach)
    ef_weights, val_weights = build_solution_weights_table(
        projected_with_structural, years, st.session_state.solutions
    )

    # 3) Diagnostic table (weights per row / year / EF or Value)
    diagnostic_df = build_diagnostic_weights_table(
        projected_with_structural, years, ef_weights, val_weights
    )

    # 4) Final attribution per solution and per year
    impact_df = compute_solution_impact_from_diagnostic(
        projected_with_structural,
        projected_with_solutions,
        df_avoided,
        diagnostic_df,
        years
    )

    # 5) Cache everything in session_state for reuse in Results + Visualisations
    st.session_state["df_emissions_before"] = df_emissions_before
    st.session_state["df_emissions_after"] = df_emissions_after
    st.session_state["df_avoided"] = df_avoided
    st.session_state["ef_weights"] = ef_weights
    st.session_state["val_weights"] = val_weights
    st.session_state["diagnostic_df"] = diagnostic_df
    st.session_state["impact_df"] = impact_df

# =========================================
# 2. Tab 1: Home
# =========================================

with tabs[0]:
    st.title("Home: Import your file")

    col1, col2 = st.columns(2)

    # -----------------------
    # Load new Excel file
    # -----------------------
    with col1:
        st.markdown("### Welcome! Please upload your Excel files first")
        st.markdown("#### You need to use the required templates to get started.")

        uploaded_file = st.file_uploader("Upload your footprint file", type=["xlsx"])
        if uploaded_file:
            try:
                data = pd.read_excel(uploaded_file)
                st.session_state["data"] = data

                required_columns = ["Category", "Sub-category 1", "Name", "Emissions"]
                missing_cols = [col for col in required_columns if col not in data.columns]

                if missing_cols:
                    st.error(f"The following required columns are missing: {', '.join(missing_cols)}")
                elif data[["Category", "Sub-category 1"]].isnull().any().any():
                    st.error("Some rows have missing values in 'Category' or 'Sub-category 1'. Please fix them.")
                else:
                    st.success("File uploaded and structure validated!")

            except Exception as e:
                st.error(f"Error while reading the file: {e}")

    # -----------------------
    # Load saved session (JSON)
    # -----------------------
    with col2:
        st.markdown("### Load a previously saved session")
        st.markdown("#### If you have already used the app and saved a file")

        saved_session = st.file_uploader(
            "Upload your saved session (.json)",
            type=["json"],
            key="json_loader"
        )

        if saved_session:
            import copy, hashlib

            # Compute a stable fingerprint of the uploaded file contents
            file_bytes = saved_session.getvalue()
            digest = hashlib.md5(file_bytes).hexdigest()

            # Only load if this exact file hasn’t been loaded yet
            if st.session_state.get("_loaded_json_digest") != digest:
                try:
                    # --- 1) Parse JSON from bytes (avoid re-reading the file object)
                    session_data = json.loads(file_bytes.decode("utf-8"))

                    # --- 2) Restore base keys
                    for key, value in session_data.items():
                        st.session_state[key] = value

                    # --- 3) Rebuild DataFrame if it was saved as dict
                    if "data_dict" in st.session_state:
                        st.session_state["data"] = pd.DataFrame.from_dict(
                            st.session_state.pop("data_dict")
                        )

                    # --- 4) Deep-copy nested objects so Streamlit tracks mutations
                    for key in ["solutions", "growth_inputs", "structural_effects"]:
                        if key in st.session_state and isinstance(st.session_state[key], (list, dict)):
                            st.session_state[key] = copy.deepcopy(st.session_state[key])

                    # --- 5) Reactivate mutable lists (critical for adding new items later)
                    for key in ["solutions", "growth_inputs", "structural_effects"]:
                        if key in st.session_state and isinstance(st.session_state[key], list):
                            st.session_state[key] = list(st.session_state[key])

                    # ✅ Do NOT rebuild any solutions_table/targets_table mirrors here.
                    # The single source of truth is inside each solution.

                    # Mark this exact file as loaded so it won’t reload on subsequent reruns
                    st.session_state["_loaded_json_digest"] = digest

                    st.success("✅ Session restored successfully! You can now go to the other tabs.")

                    # Optional: immediately refresh UI after one-time load
                    st.rerun()

                except Exception as e:
                    st.error(f"Could not load session: {e}")
            else:
                # Same file already loaded; do nothing (prevents overwriting new changes on reruns)
                st.info("This session file is already loaded. Edit freely; your changes won’t be overwritten.")

    # -----------------------
    # Quick preview & basic setup
    # -----------------------
    st.header("Now let's visualize what we have!")
    col3, col4 = st.columns(2)

    with col3:
        if "data" in st.session_state:
            data = st.session_state["data"]
            st.write("### ✅ Data preview", data.head())

    with col4:
        if "data" in st.session_state:
            # Determine default years
            if "years" in st.session_state and st.session_state["years"]:
                default_start = min(st.session_state["years"])
                default_end = max(st.session_state["years"])
            else:
                default_start = 2025
                default_end = 2035

            # Let user define projection range
            start_year = st.sidebar.number_input("Start year", value=default_start, step=1)
            end_year = st.sidebar.number_input("End year", value=default_end, step=1, min_value=start_year)

            # Save in session state
            st.session_state["years"] = list(range(start_year, end_year + 1))

            # Let user define the colors & basic charts
            choose_colors(data["Category"].unique())
            show_pie_chart_by_category(data)
            show_total_emissions(data)
            # If needed later: build_tree(data) is provided by modules.tree


# =========================================
# 3. Tab 2: Growth
# =========================================

with tabs[1]:
    st.title("Growth Projections")

    if has_loaded_data():
        data = st.session_state["data"]
        years = st.session_state["years"]

        # 🌍 STEP 1 — Define inflation (must come before growth projections)
        with st.expander("📈 Define inflation assumptions", expanded=False):
            define_inflation(years)

        # 📊 STEP 2 — Create growth or budget projections
        with st.expander("➕ Create a new growth or budget projection", expanded=True):
            create_growth(years)

        # 🧭 STEP 3 — Assign projections to categories
        st.markdown("## 📌 Assign projections to categories")
        assign_growth(data)

        summarize_growths(years)

        # 🧮 STEP 4 — Apply projections to data (adjusted for inflation)
        st.header("Projected Values (real terms)")
        base_projection = create_projection_base(data, years)
        projected = apply_projections_to_base(base_projection, years)

        # 🧩 STEP 5 — Check consistency and display results
        check_projection_coverage(projected)
        st.session_state["projected"] = projected
        # Save growth-only version (without structural effects)
        st.session_state["projected_growth_only"] = projected.copy()

        st.dataframe(projected, use_container_width=True)

    else:
        st.info("Please upload a dataset in the Home tab first.")


# =========================================
# 4. Tab 3: Structural Effects
# =========================================

with tabs[2]:
    st.title("Structural Effects")

    if has_loaded_data():
        data = st.session_state["data"]
        years = st.session_state["years"]
        projected = st.session_state.get("projected")

        init_structural_effects()
        create_structural_effect()
        assign_structural_effects(data)

        projected_with_structural = apply_structural_effects(projected)
        st.session_state["projected_with_structural"] = projected_with_structural

        # Compute structural effect impact once and store it
        structural_impact = compute_structural_impact(
            st.session_state["projected_growth_only"],
            st.session_state["years"]
        )
        st.session_state["structural_effects_impact"] = structural_impact

        check_structural_coverage(projected_with_structural)
        st.dataframe(projected_with_structural, use_container_width=True)

    else:
        st.info("Please upload a dataset in the Home tab first.")


# =========================================
# 5. Tab 4: Solutions
# =========================================

with tabs[3]:
    st.title("Solutions")

    if has_loaded_data():
        data = st.session_state["data"]
        years = st.session_state["years"]
        projected_with_structural = st.session_state.get("projected_with_structural")

        # -----------------------
        # Init + one-time migration of legacy mirrors (if any)
        # -----------------------
        init_solutions()

        def _migrate_legacy_targets_into_solutions():
            """
            One-time migration: if old mirrors (solutions_table / targets_table) exist,
            push their content into solutions[i]['years_targets'] and drop the mirrors.
            Uses solution 'id' when possible, otherwise falls back to 'name'.

            Rationale:
            - Historically, we mirrored target years in separate dicts keyed by name.
            - This caused drift between UI-edited values and exported/restored values.
            - We now enforce a single source of truth: inside each solution.
            """
            sols = st.session_state.get("solutions", [])
            mirrors = {}

            # Collect any legacy mirrors
            if "solutions_table" in st.session_state and isinstance(st.session_state["solutions_table"], dict):
                mirrors.update(st.session_state["solutions_table"])
            if "targets_table" in st.session_state and isinstance(st.session_state["targets_table"], dict):
                # Prefer solutions_table if both exist; targets_table only fills missing
                for k, v in st.session_state["targets_table"].items():
                    mirrors.setdefault(k, v)

            if not mirrors or not sols:
                return

            # Build lookup by id and by name
            by_id = {s.get("id"): s for s in sols if s.get("id")}
            by_name = {s.get("name"): s for s in sols if s.get("name")}

            # Try id keys first; then name keys
            for key, ytargets in list(mirrors.items()):
                target_obj = None
                if key in by_id:
                    target_obj = by_id[key]
                elif key in by_name:
                    target_obj = by_name[key]

                if target_obj is not None and isinstance(ytargets, dict):
                    # Keep existing keys, overwrite with mirror values (as floats in 0..1)
                    merged = dict(target_obj.get("years_targets", {}))
                    for y, v in ytargets.items():
                        try:
                            merged[str(int(y))] = float(v)
                        except Exception:
                            pass
                    target_obj["years_targets"] = merged

            # Remove mirrors to avoid future drift
            for k in ("solutions_table", "targets_table"):
                if k in st.session_state:
                    del st.session_state[k]

        # Perform migration once on load of this tab
        _migrate_legacy_targets_into_solutions()

        # -----------------------
        # Create + Configure + Apply
        # -----------------------
        create_solution()
        select_solution(data, years)

        # Apply solutions (simple + mixed) to the structural scenario
        projected_with_solutions = apply_solutions(projected_with_structural, years)
        st.session_state["projected_with_solutions"] = projected_with_solutions

    else:
        st.info("Please upload your footprint file in the Home tab.")


# =========================================
# 6. Tab 5: Results
# =========================================

with tabs[4]:
    st.title("📊 Results")

    if has_loaded_data():
        years = st.session_state["years"]
        projected_with_structural = st.session_state.get("projected_with_structural")
        projected_with_solutions = st.session_state.get("projected_with_solutions")

        # -----------------------------------------------------
        # 🚀 Manual trigger to run / refresh all heavy results
        # -----------------------------------------------------
        st.markdown("### ⚙️ Run / refresh results computation")
        if st.button("🚀 Run / refresh results", key="run_results_button"):
            run_results_computation()

        # Retrieve cached outputs (if any)
        impact_df = st.session_state.get("impact_df", pd.DataFrame())
        df_emissions_before = st.session_state.get("df_emissions_before", pd.DataFrame())
        df_emissions_after = st.session_state.get("df_emissions_after", pd.DataFrame())
        df_avoided = st.session_state.get("df_avoided", pd.DataFrame())
        ef_weights = st.session_state.get("ef_weights", {})
        val_weights = st.session_state.get("val_weights", {})
        diagnostic_df = st.session_state.get("diagnostic_df", pd.DataFrame())

        # If nothing cached yet, ask the user to run the computation
        if impact_df.empty or df_emissions_before.empty or df_emissions_after.empty or df_avoided.empty:
            st.info(
                "Results are not available yet. "
                "Please click **'🚀 Run / refresh results'** above after you have "
                "configured growth, structural effects, and solutions."
            )
        else:
            # -------------------------------
            # Show projected data with solutions
            # -------------------------------
            st.markdown("### Projected Data with Solutions Applied")
            st.dataframe(projected_with_solutions, use_container_width=True)

            # -------------------------------
            # Debug: weights tables
            # -------------------------------
            with st.expander("🧪 Debug – solution weights table", expanded=False):
                df_debug = build_weights_debug_table(ef_weights, val_weights, years)

                if df_debug.empty:
                    st.info("No non-zero weights found. Check your solutions configuration.")
                else:
                    # Optional: filter by solution for easier inspection
                    solutions_list = sorted(df_debug["Solution"].unique())
                    selected_solution = st.selectbox(
                        "Filter by solution",
                        options=["(All)"] + solutions_list,
                        index=0,
                    )

                    if selected_solution != "(All)":
                        df_to_show = df_debug[df_debug["Solution"] == selected_solution]
                    else:
                        df_to_show = df_debug

                    st.markdown("### Detailed weights (non-zero only)")
                    st.dataframe(df_to_show, use_container_width=True)

                    # Optional: show an aggregated summary
                    st.markdown("### Summary per solution / year / field")
                    df_summary = build_weights_summary_table(df_debug)
                    if selected_solution != "(All)":
                        df_summary = df_summary[df_summary["Solution"] == selected_solution]

                    st.dataframe(df_summary, use_container_width=True)

            # -------------------------------
            # Diagnostic table (human-readable)
            # -------------------------------
            diagnostic_df_str = diagnostic_df.applymap(
                lambda cell: ", ".join(f"{s}: {v}%" for s, v in cell) if isinstance(cell, list) else ""
            )

            # -------------------------------
            # Final impact table
            # -------------------------------
            st.markdown("### 🧮 Final attribution of emissions reduction by solution")
            st.dataframe(impact_df.style.format("{:.2f}"), use_container_width=True)

            # -------------------------------
            # Mass-balance consistency check
            # -------------------------------
            with st.expander("🔎 Consistency check – mass balance of avoided emissions", expanded=False):
                import numpy as np

                # a) Total avoided emissions per year from df_avoided
                emission_cols = [f"Emissions_{y}" for y in years if f"Emissions_{y}" in df_avoided.columns]

                if not emission_cols:
                    st.error("No emissions columns found in df_avoided – cannot run mass-balance check.")
                else:
                    avoided_per_year_raw = df_avoided[emission_cols].sum(axis=0)
                    avoided_per_year = pd.Series(
                        {y: avoided_per_year_raw.get(f"Emissions_{y}", 0.0) for y in years},
                        index=years,
                        name="Avoided emissions (before - after)"
                    )

                    # b) Total attributed to solutions per year from impact_df
                    year_col_map = {}
                    for col in impact_df.columns:
                        try:
                            col_year = int(col)
                        except (ValueError, TypeError):
                            continue
                        if col_year in years:
                            year_col_map[col_year] = col

                    if not year_col_map:
                        st.error("Could not find any year-like columns in the solution impact table – mass-balance check skipped.")
                    else:
                        common_years = sorted(year_col_map.keys())
                        solutions_per_year_raw = impact_df[[year_col_map[y] for y in common_years]].sum(axis=0)
                        solutions_per_year = pd.Series(
                            {y: solutions_per_year_raw[year_col_map[y]] for y in common_years},
                            index=common_years,
                            name="Attributed to solutions"
                        )

                        avoided_aligned = avoided_per_year.reindex(common_years)

                        check_df = pd.concat([avoided_aligned, solutions_per_year], axis=1)
                        check_df["Residual"] = check_df["Attributed to solutions"] - check_df["Avoided emissions (before - after)"]

                        total_avoided = avoided_aligned.sum()
                        total_attributed = solutions_per_year.sum()
                        total_residual = total_attributed - total_avoided

                        st.markdown("#### Per-year comparison")
                        st.dataframe(check_df.style.format("{:.4f}"), use_container_width=True)

                        st.markdown("#### Global mass-balance check")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total avoided emissions (before - after)", f"{total_avoided:,.4f}")
                        with col_b:
                            st.metric("Total attributed to solutions", f"{total_attributed:,.4f}")
                        with col_c:
                            st.metric("Residual (attributed - avoided)", f"{total_residual:,.4f}")

                        atol = max(1e-6, abs(total_avoided) * 1e-3)
                        if abs(total_residual) <= atol:
                            st.success(
                                "✅ Mass balance OK: the sum of solution impacts is consistent with "
                                "the difference between total emissions before and after solutions "
                                f"(|residual| ≤ {atol:.4f})."
                            )
                        else:
                            st.warning(
                                "⚠️ Mass balance mismatch detected: the sum of impacts attributed to solutions "
                                "does not perfectly match the total avoided emissions. This may indicate a bug "
                                "in the attribution logic or in the weight calculations."
                            )
                            st.caption(
                                "Tip: check the debug weights table above, especially for mixed solutions, "
                                "to see which rows/years receive unexpected weights."
                            )

    else:
        st.info("Please upload your footprint file in the Home tab.")


# =========================================
# 7. Tab 6: Visualisations
# =========================================



with tabs[5]:
    st.title("📊 Visualisations")
    
    # Optional: allow recomputing results directly from this tab
    st.markdown("### ⚙️ Run / refresh results for visualisations")
    if st.button("🚀 Run / refresh results & charts", key="run_results_from_visu"):
        run_results_computation()

    # Retrieve what we need from session (guard usage)
    impact_df = st.session_state.get("impact_df", pd.DataFrame())
    df_emissions_before = st.session_state.get("df_emissions_before", pd.DataFrame())

    if has_loaded_data() and not impact_df.empty and not df_emissions_before.empty:

        years = st.session_state["years"]
        last_year = max(years)  # 🔚 We focus on the last projection year

        # ---------------------------------------------------------
        # 🧾 Emissions overview for last year (growth / structural / solutions)
        # ---------------------------------------------------------
        projected_growth_only = st.session_state.get("projected_growth_only")
        projected_with_structural = st.session_state.get("projected_with_structural")
        projected_with_solutions = st.session_state.get("projected_with_solutions")
        data_baseline = st.session_state.get("data")

        if (
            projected_growth_only is not None
            and projected_with_structural is not None
            and projected_with_solutions is not None
            and data_baseline is not None
        ):
        

            # 1) Compute emissions per scenario and per year
            df_em_growth = compute_emissions_per_year(projected_growth_only, years)
            df_em_struct = compute_emissions_per_year(projected_with_structural, years)
            df_em_solutions = compute_emissions_per_year(projected_with_solutions, years)

            # 2) Extract total emissions for the last year
            col_last = f"Emissions_{last_year}"

            growth_only_total = df_em_growth[col_last].sum()
            structural_total = df_em_struct[col_last].sum()
            solutions_total = df_em_solutions[col_last].sum()

            # 3) Total tonnes of CO₂ saved by solutions in the last year
            saved_by_solutions_last_year = structural_total - solutions_total

            # 4) Baseline emissions (initial footprint, from imported file)
            #    We assume the input file contains a column 'Emissions'
            #    with the baseline inventory (single year or average).
            if "Emissions" in data_baseline.columns:
                baseline_total = data_baseline["Emissions"].sum()
            else:
                baseline_total = np.nan

            # 5) Percentage reductions
            #    (we protect against division by zero or missing data)
            if growth_only_total and growth_only_total != 0:
                reduction_vs_growth_only_pct = (growth_only_total - solutions_total) / growth_only_total * 100.0
            else:
                reduction_vs_growth_only_pct = np.nan

            if baseline_total and baseline_total != 0:
                reduction_vs_baseline_pct = (baseline_total - solutions_total) / baseline_total * 100.0
            else:
                reduction_vs_baseline_pct = np.nan

            # 6) Build a summary table
            #    We keep two columns:
            #      - "Emissions (tCO₂e)" for levels
            #      - "Reduction (%)" for percentage indicators
            #    Some rows will only have emissions, others only percentages.
            summary_last_year = pd.DataFrame(
                {
                    "Scenario": [
                        f"Growth only ({last_year})",
                        f"Growth + structural effects ({last_year})",
                        f"Growth + structural + solutions ({last_year})",
                        f"Saved by solutions ({last_year})",
                    ],
                    "Emissions (tCO₂e)": [
                        growth_only_total,
                        structural_total,
                        solutions_total,
                        saved_by_solutions_last_year,
                    ],
                    "Reduction (%)": [np.nan, np.nan, np.nan, np.nan],
                }
            )

            # Add two extra rows for percentage reductions
            extra_rows = pd.DataFrame(
                {
                    "Scenario": [
                        f"Reduction vs growth-only projection ({last_year})",
                        "Reduction vs baseline footprint",
                    ],
                    "Emissions (tCO₂e)": [np.nan, np.nan],
                    "Reduction (%)": [
                        reduction_vs_growth_only_pct,
                        reduction_vs_baseline_pct,
                    ],
                }
            )

            summary_last_year = pd.concat([summary_last_year, extra_rows], ignore_index=True)

            st.markdown(f"### 🧾 Emissions overview for {last_year}")
            st.dataframe(
                summary_last_year.style.format(
                    {
                        "Emissions (tCO₂e)": "{:,.0f}",
                        "Reduction (%)": "{:.1f}%"
                    }
                ),
                use_container_width=True,
            )

        else:
            st.info(
                "Some intermediate scenarios are missing (growth-only / structural / solutions / baseline). "
                "Please complete the previous tabs to enable the last-year overview table."
            )

        # =========================================================
        # --- CONFIGURATION ---
        # =========================================================
        st.markdown("### ⚙️ Visualisation settings")

        include_structural = st.toggle(
            "Include structural effects in 'No action' scenario",
            value=True,
            help="If disabled, structural effects appear as a first solution."
        )

                # =========================================================
        # --- HANDLE STRUCTURAL EFFECTS TOGGLE ---
        # =========================================================
        if include_structural:
            # ✅ Structural effects already included in the 'No action' scenario
            # Here, df_emissions_before is based on projected_with_structural
            df_emissions_base = df_emissions_before.copy()

        else:
            # ❌ Structural effects are displayed as a separate "solution"
            struct_impact = st.session_state.get("structural_effects_impact")
            projected_growth_only = st.session_state.get("projected_growth_only")

            if struct_impact is not None:
                # Prepare a clean DataFrame with unique index
                struct_impact_df = pd.DataFrame([struct_impact])
                struct_impact_df.index = ["Structural effects"]

                # Remove any existing duplicate entry
                impact_df = impact_df.drop(index="Structural effects", errors="ignore")
                impact_df = impact_df.loc[~impact_df.index.duplicated(keep="first")]

                # Concatenate safely (Structural effects appear as first "solution")
                impact_df = pd.concat([struct_impact_df, impact_df])

                # Initialize color configuration if not already present
                if "solution_colors" not in st.session_state:
                    st.session_state.solution_colors = {}

                # Assign a neutral gray color if not already set
                if "Structural effects" not in st.session_state.solution_colors:
                    st.session_state.solution_colors["Structural effects"] = "#888888"

                st.info("Structural effects are displayed as a separate solution.")
            else:
                st.warning("Structural effects impact not found — please apply them in the Structural tab first.")

            # 🧠 IMPORTANT: when structural effects are separated,
            # the 'No action' scenario must be *growth only*,
            # otherwise structural effects would be counted twice.
            if projected_growth_only is not None:
                df_emissions_base = compute_emissions_per_year(
                    projected_growth_only,
                    years
                )
            else:
                # Fallback to df_emissions_before if something is missing,
                # but warn the user because this is conceptually inconsistent.
                st.warning(
                    "Growth-only scenario not found in session state. "
                    "Using the structural scenario as base, which may double-count structural effects."
                )
                df_emissions_base = df_emissions_before.copy()


        # =========================================================
        # --- COLORS + ORDER ---
        # =========================================================
        # Guarantee uniqueness before plotting
        impact_df = impact_df.loc[~impact_df.index.duplicated(keep="first")]

        solutions = list(impact_df.index)
        choose_solution_colors_and_order(solutions)
        solution_colors = st.session_state.solution_colors
        solution_order = st.session_state.solution_order

        # =========================================================
        # --- COMPUTE SHARES AND PLOTS ---
        # =========================================================
        impact_df = compute_solution_percentages(impact_df, df_emissions_base)
        impact_df = impact_df.loc[solution_order]
        total_reduction = impact_df["Total"].sum()

        st.markdown(f"**🌍 Total avoided emissions: {total_reduction:,.0f} tCO₂e**")

        # =========================================================
        # --- PLOTS ---
        # =========================================================
        col1, col2 = st.columns(2)

        # === Left column : cumulative ===
        with col1:
            st.markdown("#### 📈 Cumulative reductions")
            fig_cumulate = plot_cumulative_emissions_reduction(
                df_emissions_base, impact_df, solution_colors, True
            )
            st.pyplot(fig_cumulate)
            st.download_button(
                "⬇️ Download SVG",
                export_svg(fig_cumulate, "cumulative.svg"),
                file_name="cumulative.svg",
                mime="image/svg+xml",
            )

        # === Right column : annual + waterfall ===
        with col2:
            st.markdown("#### 📆 Annual avoided emissions")
            fig_annual = plot_annual_emissions_reduction(
                df_emissions_base, impact_df, solution_colors, True
            )
            st.pyplot(fig_annual)
            st.download_button(
                "⬇️ Download SVG",
                export_svg(fig_annual, "annual.svg"),
                file_name="annual.svg",
                mime="image/svg+xml",
            )

            st.markdown("#### 💧 Waterfall of emission reductions")
            start_value, steps, labels, colors = prepare_waterfall_inputs(
                df_emissions_base, impact_df, solution_colors
            )
            fig_waterfall = plot_waterfall_emissions(start_value, steps, labels, colors)
            st.pyplot(fig_waterfall)
            st.download_button(
                "⬇️ Download SVG",
                export_svg(fig_waterfall, "waterfall.svg"),
                file_name="waterfall.svg",
                mime="image/svg+xml",
            )

    else:
        st.info("Please upload a dataset first and compute results in the Results tab.")



# =========================================
# 8. Tab 7: Export
# =========================================

with tabs[6]:
    st.markdown("## 💾 Save your work")

    if has_loaded_data():
        # -----------------------
        # Choose export file name
        # -----------------------
        file_name = st.text_input(
            "Choose a name for your session file (without extension)",
            value="carbon_session"
        )

        # -----------------------
        # Define keys to include
        # -----------------------
        keys_to_save = [
            'solutions',
            'growth_inputs',
            'structural_effects',
            'growth_assignments',
            'structural_assignments',
            'category_colors',
            'solution_colors'
        ]

        # -----------------------
        # Copy session data to export dictionary
        # -----------------------
        session_to_export = {
            k: st.session_state[k]
            for k in keys_to_save
            if k in st.session_state
        }

        # -----------------------
        # ✅ Solutions already carry the latest years_targets
        # -----------------------
        session_to_export["solutions"] = st.session_state.get("solutions", [])

        # -----------------------
        # Store data and years
        # -----------------------
        session_to_export["data_dict"] = st.session_state["data"].to_dict()
        session_to_export["years"] = st.session_state["years"]

        # -----------------------
        # Export as JSON
        # -----------------------
        json_bytes = json.dumps(session_to_export, indent=2).encode("utf-8")
        buffer = BytesIO(json_bytes)

        st.download_button(
            label="📥 Download session as JSON",
            data=buffer,
            file_name=f"{file_name}.json",
            mime="application/json"
        )

        # -----------------------
        # Optional Excel export (for human-readable review)
        # -----------------------
        if st.button("📈 Export solutions & growths as Excel"):
            with pd.ExcelWriter("carbon_export.xlsx") as writer:
                # Convert each structure to Excel sheet if available
                if "solutions" in st.session_state:
                    df_solutions = pd.DataFrame(st.session_state["solutions"])
                    df_solutions.to_excel(writer, sheet_name="Solutions", index=False)

                if "growth_inputs" in st.session_state:
                    pd.DataFrame(st.session_state["growth_inputs"]).to_excel(writer, sheet_name="Growth Inputs", index=False)

                if "structural_effects" in st.session_state:
                    pd.DataFrame(st.session_state["structural_effects"]).to_excel(writer, sheet_name="Structural Effects", index=False)

                if "growth_assignments" in st.session_state:
                    pd.DataFrame(st.session_state["growth_assignments"]).to_excel(writer, sheet_name="Growth Assignments", index=False)

                if "structural_assignments" in st.session_state:
                    pd.DataFrame(st.session_state["structural_assignments"]).to_excel(writer, sheet_name="Structural Assignments", index=False)

            # Read Excel file into memory
            with open("carbon_export.xlsx", "rb") as f:
                excel_bytes = f.read()

            st.download_button(
                label="📥 Download Excel",
                data=excel_bytes,
                file_name="carbon_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.info("You need to upload or restore a dataset before saving.")
