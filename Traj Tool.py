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

#- [ ] Review code and comment properly
#- [ ] Display nice charts
#- [ ] Clarify databefore and dataafter to calculate the reduction with growth and structural effects differents roles (voir avec Paolo)


## Done
#- [X] Enable input of growth forecasts (with multiple possible growth scenarios); determine how to assign growth to categories/subcategories
#- [X] Handle import of new input format and allow simple visualisation by category
#- [X] Enable input of structural effects and manage their assignment
#- [X] Allow users to create solutions (simple and mix )
#- [X] Calculate the emissions after the solutions attribution
#- [X] Display the impact of each solution
#- [X] Display projected values by name and year
#- [X] Allow export of a file to avoid starting from scratch


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

from modules.colors import choose_colors, show_pie_chart_by_category
from modules.tree import create_projection_base
from modules.growth import create_growth, assign_growth , apply_projections_to_base, check_projection_coverage
from modules.structural import init_structural_effects, create_structural_effect , assign_structural_effects, apply_structural_effects, check_structural_coverage
from modules.solutions import init_solutions, select_solution, apply_solutions, create_solution, compute_avoided_emissions, compute_emissions_per_year
from modules.solutions import build_diagnostic_weights_table, build_solution_weights_table, compute_solution_impact_from_diagnostic
from modules.visualisation import plot_cumulative_emissions_reduction, plot_annual_emissions_reduction

# Activate wide layout mode to reduce side margins (must be the first Streamlit command)
st.set_page_config(layout="wide")

# Main tabs
tabs = st.tabs(["Home", "Growth", "Structural Effects", "Solutions", "Visualisations","Export"])

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
        st.markdown("#### If you have allready used the app and saved a file")
    
        saved_session = st.file_uploader("Upload your saved session (.json)", type=["json"], key="json_loader")
        if saved_session:
            try:
                session_data = json.load(saved_session)
    
                for key, value in session_data.items():
                    st.session_state[key] = value
    
                # Rebuild DataFrame from stored dict
                if "data_dict" in st.session_state:
                    st.session_state["data"] = pd.DataFrame.from_dict(st.session_state.pop("data_dict"))
    
                st.success("Session restored! You can now go to the other tabs.")
    
            except Exception as e:
                st.error(f"Could not load session: {e}")

    
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
            #build_tree(data) + also need to import this fonction from the module tree


# =========================================
# Tab 2: Growth
# =========================================

with tabs[1]:
    st.title("Growth Projections")

    if has_loaded_data():
        data = st.session_state["data"]
        years = st.session_state["years"]

        with st.expander("➕ Create a new growth or budget projection", expanded=True):
            create_growth(years)

        st.markdown("## 📌 Assign projections to categories")
        assign_growth(data)
        
        st.header("Projected Values")

        base_projection = create_projection_base(data, years)
        projected = apply_projections_to_base(base_projection, years)

        check_projection_coverage(projected)
        st.session_state["projected"] = projected

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


        projected = apply_structural_effects(projected)
        st.session_state["projected"] = projected

        check_structural_coverage(projected)
        st.dataframe(projected, use_container_width=True)
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

        init_solutions()
        select_solution(data, years)
        create_solution()

        projected_with_solutions = apply_solutions(projected, years)
        st.markdown("### Projected Data with Solutions Applied")
        st.dataframe(projected_with_solutions, use_container_width=True)

        df_emissions_before = compute_emissions_per_year(projected, years)
        df_emissions_after = compute_emissions_per_year(projected_with_solutions, years)
        df_avoided = compute_avoided_emissions(df_emissions_before, df_emissions_after, years)
        
        # =========================================

        # If you want to display some table
        
        st.markdown("### 🔢 Emissions BEFORE solutions")
        st.dataframe(df_emissions_before[[f"Emissions_{y}" for y in years]], use_container_width=True)
        df_only_emissions_before = df_emissions_before[[f"Emissions_{y}" for y in years]]
        st.write(df_only_emissions_before.dtypes)
        st.write(df_only_emissions_before.head())


        #st.markdown("### 🔢 Emissions AFTER solutions")
        #st.dataframe(df_emissions_after[[f"Emissions_{y}" for y in years]], use_container_width=True)

        #st.markdown("### ♻️ Avoided emissions")
        #st.dataframe(df_avoided[[f"Emissions_{y}" for y in years]].style.format("{:.2f}"), use_container_width=True) 
        
        
        ef_weights, val_weights = build_solution_weights_table(projected, years, st.session_state.solutions)
        diagnostic_df = build_diagnostic_weights_table(projected, years, ef_weights, val_weights)
        diagnostic_df_str = diagnostic_df.applymap(lambda cell: ", ".join(f"{s}: {v}%" for s, v in cell) if isinstance(cell, list) else "")

        #st.markdown("### 📊 Diagnostic of solution weights")
        #st.dataframe(diagnostic_df_str, use_container_width=True)

        impact_df = compute_solution_impact_from_diagnostic(projected,projected_with_solutions,df_avoided,diagnostic_df,years)
        st.markdown("### 🧮 Final attribution of emissions reduction by solution")
        st.dataframe(impact_df.style.format("{:.2f}"), use_container_width=True)

    else:
        st.info("Please upload your footprint file in the Home tab.")

# =========================================
# Tab 5: Visualisations
# =========================================

with tabs[4]:
    st.title("Visualisations")
    
    if has_loaded_data():
        solution_colors = {
    'Privilégier des fournisseurs green': '#00bfc4',
    'Réduction des achats': '#f8766d',
    'Change Avion to Train': '#7cae00'
}
        
        fig = plot_cumulative_emissions_reduction(
            emissions_before_df=df_emissions_before,
            reductions_by_solution_df=impact_df,
            solution_colors=solution_colors,
            show_percentage_annotation=True
        )
        
        st.pyplot(fig)
        
        st.header("📉 Annual CO2e Emissions with and without Actions")

        # Call the plotting function
        fig_annual = plot_annual_emissions_reduction(
            emissions_before_df=df_emissions_before,
            reductions_by_solution_df=impact_df,
            solution_colors=solution_colors,
            show_percentage_annotation=True  # Optional
            )
        
        st.pyplot(fig_annual)



# =========================================
# Tab 6: Export
# =========================================

with tabs[5]:
    st.markdown("## 💾 Save your work")

    if has_loaded_data():
        # To choose the export file's name
        file_name = st.text_input("Choose a name for your session file (without extension)", value="carbon_session")

        keys_to_save = ['solutions','growth_inputs','structural_effects','growth_assignments','structural_assignments','category_colors']

        session_to_export = {k: st.session_state[k] for k in keys_to_save if k in st.session_state}

        # Store data and years
        session_to_export["data_dict"] = st.session_state["data"].to_dict()
        session_to_export["years"] = st.session_state["years"]

        json_bytes = json.dumps(session_to_export, indent=2).encode('utf-8')
        buffer = BytesIO(json_bytes)

        st.download_button(label="📥 Download session as JSON",data=buffer,file_name=f"{file_name}.json", mime="application/json")
    else:
        st.info("You need to upload or restore a dataset before saving.")
       