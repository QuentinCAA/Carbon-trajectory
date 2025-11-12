# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 17:12:48 2025

@author: quent
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:47:39 2025

@author: quent
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
# 6. Tab 5: Target Dashboard      -> 
# ================================


# =========================================
# Development Notes (update every time you close the file)
# =========================================

## To Do


#- [ ] Display nice charts
#- [ ] Give the possibility to export data for financial trajectory (tableau excel avec growths et solutions avec ttes les infos disponibles pour les deux)

#- [ ] Create an utilisation guide (with screenshots) and buy doing so also facilitate the user experience with more explanations at the start of each "onglet"
#- [ ] Create some examples json to serve as example and be available to train/understand in the previous guide

#- [ ] Clarify databefore and dataafter to calculate the reduction with growth and structural effects differents roles (voir avec Paolo)

#- [] Create a link between inflation and localisation ? 


## Done
#- [X] Enable input of growth forecasts (with multiple possible growth scenarios); determine how to assign growth to categories/subcategories
#- [X] Handle import of new input format and allow simple visualisation by category
#- [X] Enable input of structural effects and manage their assignment
#- [X] Allow users to create solutions (simple and mix )
#- [X] Calculate the emissions after the solutions attribution
#- [X] Display the impact of each solution
#- [X] Display projected values by name and year
#- [X] Allow export of a file to avoid starting from scratch
#- [X] Review code and comment properly

#- [X]Integrate inflation


## Futur improvement

#- create export for the financial trajectory or integrate the financial tarjectory in the app ? 
#- create a simplier version for program manager ? (not sure really needed because allready quite easy to use I believe)

# =========================================
# 1. Code Initialization
# =========================================

import streamlit as st
import pandas as pd
import json
from io import BytesIO

from modules.colors import choose_colors, show_pie_chart_by_category, show_total_emissions
from modules.tree import create_projection_base
from modules.growth import create_growth, assign_growth , apply_projections_to_base, check_projection_coverage, define_inflation, summarize_growths
from modules.structural import init_structural_effects, create_structural_effect , assign_structural_effects, apply_structural_effects, check_structural_coverage, compute_structural_impact
from modules.solutions import init_solutions, select_solution, apply_solutions, create_solution, compute_avoided_emissions, compute_emissions_per_year
from modules.solutions import build_diagnostic_weights_table, build_solution_weights_table, compute_solution_impact_from_diagnostic
from modules.visualisation import choose_solution_colors_and_order ,plot_cumulative_emissions_reduction, plot_annual_emissions_reduction, prepare_waterfall_inputs
from modules.visualisation import plot_waterfall_emissions, export_svg, compute_solution_percentages

# Activate wide layout mode to reduce side margins (must be the first Streamlit command)
st.set_page_config(layout="wide")

# Main tabs
tabs = st.tabs(["Home", "Growth", "Structural Effects", "Solutions", "Results","Visualisations","Export"])

# Helper: check if session is ready
def has_loaded_data():
    return "data" in st.session_state and "years" in st.session_state

# ========================================= 
# Tab 1: Home
# =========================================

with tabs[0]:
    
    st.title("Home: Import your file")

    col1, col2 = st.columns(2)
    
    
    with col1:
        # ============
        # Load new Excel file
        # ============
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
    
    with col2:
        # ============  
        # Load saved session (JSON)
        # ============  
        st.markdown("### Load a previously saved session")
        st.markdown("#### If you have already used the app and saved a file")
    
        saved_session = st.file_uploader(
            "Upload your saved session (.json)",
            type=["json"],
            key="json_loader"
        )
    
        if saved_session:
            import json, pandas as pd, copy, hashlib
    
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
    
                    # --- 6) Initialize/refresh the central table of yearly targets (if you use it elsewhere)
                    if "solutions" in st.session_state:
                        if "solutions_table" not in st.session_state:
                            st.session_state.solutions_table = {}
                        # Only add missing entries; do NOT wipe existing table on reruns
                        for s in st.session_state.solutions:
                            name = s.get("name")
                            if name and name not in st.session_state.solutions_table:
                                st.session_state.solutions_table[name] = s.get("years_targets", {})
    
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




    
    # Back to full-width layout
    st.header("Now let's visualize what we have!")
    # ============
    # Display data, chart and tree if available
    # ============
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
            
            # Let user define the colors
            choose_colors(data["Category"].unique())
            show_pie_chart_by_category(data)
            show_total_emissions(data)
            #build_tree(data) + also need to import this fonction from the module tree


# =========================================
# Tab 2: Growth
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
# Tab 3: Structural Effects
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
# Tab 4: Solutions
# =========================================

with tabs[3]:
    st.title("Solutions")

    if has_loaded_data():
        data = st.session_state["data"]
        years = st.session_state["years"]
        projected = st.session_state.get("projected")
        
        # === Initialize and display the central Targets Table ===

        import pandas as pd
        import streamlit as st
        
        init_solutions()
        create_solution()
        
        # Initialize the table once per session
        if "targets_table" not in st.session_state:
            st.session_state.targets_table = {
                s["name"]: s.get("years_targets", {}) 
                for s in st.session_state.get("solutions", [])
            }
               
        select_solution(data, years)
        projected_with_solutions = apply_solutions(projected_with_structural, years)

       
    

    else:
        st.info("Please upload your footprint file in the Home tab.")

# =========================================
# Tab 5: Results
# =========================================

with tabs[4]:
    st.title("📊 Results")
    
    if has_loaded_data():
        
        st.markdown("### Projected Data with Solutions Applied")
        st.dataframe(projected_with_solutions, use_container_width=True)

        df_emissions_before = compute_emissions_per_year(projected_with_structural, years)
        df_emissions_after = compute_emissions_per_year(projected_with_solutions, years)
        df_avoided = compute_avoided_emissions(df_emissions_before, df_emissions_after, years)
        
        # =========================================

        # If you want to display some table
        
        #st.markdown("### 🔢 Emissions BEFORE solutions")
        #st.dataframe(df_emissions_before[[f"Emissions_{y}" for y in years]], use_container_width=True)
        df_only_emissions_before = df_emissions_before[[f"Emissions_{y}" for y in years]]
        #st.write(df_only_emissions_before.dtypes)
        #st.write(df_only_emissions_before.head())


        #st.markdown("### 🔢 Emissions AFTER solutions")
        #st.dataframe(df_emissions_after[[f"Emissions_{y}" for y in years]], use_container_width=True)

        #st.markdown("### ♻️ Avoided emissions")
        #st.dataframe(df_avoided[[f"Emissions_{y}" for y in years]].style.format("{:.2f}"), use_container_width=True) 
        
        
        ef_weights, val_weights = build_solution_weights_table(projected_with_structural, years, st.session_state.solutions)
        diagnostic_df = build_diagnostic_weights_table(projected_with_structural, years, ef_weights, val_weights)
        diagnostic_df_str = diagnostic_df.applymap(lambda cell: ", ".join(f"{s}: {v}%" for s, v in cell) if isinstance(cell, list) else "")

        #st.markdown("### 📊 Diagnostic of solution weights")
        #st.dataframe(diagnostic_df_str, use_container_width=True)

        impact_df = compute_solution_impact_from_diagnostic(projected_with_structural,projected_with_solutions,df_avoided,diagnostic_df,years)
        st.markdown("### 🧮 Final attribution of emissions reduction by solution")
        st.dataframe(impact_df.style.format("{:.2f}"), use_container_width=True)
        
    else:
        st.info("Please upload your footprint file in the Home tab.")

# =========================================
# Tab 6: Visualisations
# =========================================

with tabs[5]:
    st.title("📊 Visualisations")

    if has_loaded_data() and not impact_df.empty:
    
        # =========================================================
        # --- CONFIGURATION ---
        # =========================================================
        st.markdown("### ⚙️ Visualisation settings")
    
        include_structural = st.toggle(
            "Include structural effects in 'No action' scenario",
            value=True,
            help="If disabled, structural effects appear as a first solution."
        )
    
        years = st.session_state["years"]
    
        # =========================================================
        # --- HANDLE STRUCTURAL EFFECTS TOGGLE ---
        # =========================================================
        if include_structural:
            # ✅ Structural effects already included in EF baseline
            df_emissions_base = df_emissions_before.copy()

        else:
            # ❌ Structural effects shown as a separate solution
            struct_impact = st.session_state.get("structural_effects_impact")

            if struct_impact is not None:
                # Prepare a clean DataFrame with unique index
                struct_impact_df = pd.DataFrame([struct_impact])
                struct_impact_df.index = ["Structural effects"]

                # Remove any existing duplicate entry
                impact_df = impact_df.drop(index="Structural effects", errors="ignore")
                impact_df = impact_df.loc[~impact_df.index.duplicated(keep="first")]

                # Concatenate safely
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
        st.info("Please upload a dataset first.")




# =========================================
# Tab 7: Export
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
        # 💡 Integrate solutions_table before saving
        # -----------------------
        if "solutions" in st.session_state and "solutions_table" in st.session_state:
            merged_solutions = []
            for s in st.session_state["solutions"]:
                s_copy = s.copy()
                s_copy["years_targets"] = st.session_state["solutions_table"].get(s["name"], {})
                merged_solutions.append(s_copy)
            session_to_export["solutions"] = merged_solutions

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
                    # Use merged_solutions if it exists, otherwise raw
                    df_solutions = pd.DataFrame(
                        merged_solutions if "merged_solutions" in locals() else st.session_state["solutions"]
                    )
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
