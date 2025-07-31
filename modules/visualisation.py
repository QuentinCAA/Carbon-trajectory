# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:29:57 2025

@author: quent
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def plot_cumulative_emissions_reduction(
    emissions_before_df: pd.DataFrame,
    reductions_by_solution_df: pd.DataFrame,
    solution_colors: dict = None,
    show_percentage_annotation: bool = True
):
    """
    Plot a cumulative CO2e emissions graph with:
    - Emissions without action (black line)
    - Emissions trajectory with actions (red line)
    - Reductions stacked between both

    Parameters:
    - emissions_before_df: DataFrame with columns like "Emissions_2025", ..., one row per category
    - reductions_by_solution_df: DataFrame with solutions as index, years as columns (int), and values in tCO2e
    - solution_colors: optional dict of colors by solution name (e.g., {'My solution': '#FF0000'})
    - show_percentage_annotation: if True, displays the % reduction in final year

    Returns:
    - fig (matplotlib.figure.Figure)
    """

    # 1. Sanitize emissions_before_df
    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[[col for col in emissions_before_df.columns if col.startswith("Emissions_")]]
    emissions_before_df = emissions_before_df.applymap(lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x)
    emissions_before_df = emissions_before_df.dropna(how='all')

    # 2. Compute emissions without action (baseline)
    emissions_without_action = emissions_before_df.sum(axis=0)
    emissions_without_action.index = emissions_without_action.index.str.extract(r'Emissions_(\d+)', expand=False).astype(int)
    emissions_cumulative = emissions_without_action.cumsum()

    # DEBUG
    #st.subheader("🔍 Emissions cumulative without actions")
    #st.line_chart(emissions_cumulative)

    # 3. Prepare reductions_by_solution_df
    reductions_by_solution_df = reductions_by_solution_df.copy()
    reductions_by_solution_df.columns = reductions_by_solution_df.columns.astype(int)

    # Debug: affichage brut
    #st.subheader("🔍 Reductions by solution (raw input)")
    #st.dataframe(reductions_by_solution_df)

    reductions_by_year = reductions_by_solution_df.T
    reductions_cumulative = reductions_by_year.cumsum()

    # Debug: cumulative reductions
    # st.subheader("🔍 Reductions cumulative (by year and solution)")
    # st.dataframe(reductions_cumulative)

    # Ensure same year index
    years = reductions_cumulative.index
    emissions_cumulative = emissions_cumulative.loc[years]

    # 4. Compute trajectory = emissions_cumulative - total_reduction
    total_reduction = reductions_cumulative.sum(axis=1)
    trajectory = emissions_cumulative - total_reduction

    # Debug: trajectoire
    # st.subheader("🔍 Trajectory with actions")
    # st.line_chart(trajectory)

    # 5. Compute height of each band: bottom = trajectory, top = trajectory + each solution
    bands = []
    bottom = trajectory.copy()
    for col in reductions_cumulative.columns:
        top = bottom + reductions_cumulative[col]
        bands.append((col, bottom.copy(), top.copy()))
        
        # DEBUG: aperçu des courbes intermédiaires
        #st.write(f"📊 Band for: {col}")
        #st.line_chart(pd.DataFrame({f"bottom_{col}": bottom, f"top_{col}": top}))
        
        bottom = top

    # 6. Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each solution band
    for col, y_bottom, y_top in bands:
        color = solution_colors[col] if solution_colors and col in solution_colors else None
        ax.fill_between(years, y_bottom, y_top, label=col, color=color, alpha=0.8)

    # Curve: Emissions without action (black)
    ax.plot(emissions_cumulative.index, emissions_cumulative.values, color='black', linewidth=2, label='Emissions without action')

    # Curve: Trajectory (red)
    ax.plot(trajectory.index, trajectory.values, color='red', linewidth=2, label='Trajectory')

    

    # Aesthetics
    ax.set_title("Cumulative CO2e emissions with and without actions", fontsize=16, fontweight='bold' )
    ax.set_xlabel("Year")
    ax.set_ylabel("Tonnes of CO2e")
    ax.grid(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    
    # Hide top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove grid
    ax.grid(False)
    
    # Background colors
    fig.patch.set_facecolor('white')        # Outside figure background
    ax.set_facecolor('#FAFAFA')             # Plot area background
    
    # Legend outside
    ax.legend(title="Solutions", loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, title_fontsize=12)
    
    # Add reduction box below the legend area (right side)
    # Percentage reduction annotation
    if show_percentage_annotation:
        final_year = years.max()
        without = emissions_cumulative.loc[final_year]
        with_action = trajectory.loc[final_year]
        percent_reduction = 100 * (1 - with_action / without)

        fig.text(0.88, 0.52,  # X and Y position in figure coordinates (tune as needed)
                 f"{percent_reduction:.0f}%\nreduction\nin {final_year}",
                 fontsize=12, color='red',
                 ha='center', va='top',
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.4'))
    
    import io

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    st.download_button(
        label="📥 Download PNG",
        data=buffer.getvalue(),
        file_name="cumulative_emissions.png",
        mime="image/png"
    )

    return fig


def plot_annual_emissions_reduction(
    emissions_before_df: pd.DataFrame,
    reductions_by_solution_df: pd.DataFrame,
    solution_colors: dict = None,
    show_percentage_annotation: bool = True
):
    """
    Plot annual (non-cumulative) CO2e emissions graph with:
    - Emissions without action (black line)
    - Emissions trajectory with actions (red line)
    - Annual reductions stacked between both

    Parameters:
    - emissions_before_df: DataFrame with columns like "Emissions_2025", ..., one row per category
    - reductions_by_solution_df: DataFrame with solutions as index, years as columns (int), and values in tCO2e
    - solution_colors: optional dict of colors by solution name (e.g., {'My solution': '#FF0000'})
    - show_percentage_annotation: if True, displays the % reduction in final year

    Returns:
    - fig (matplotlib.figure.Figure)
    """

    # 1. Clean and prepare emissions_before_df
    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[[col for col in emissions_before_df.columns if col.startswith("Emissions_")]]
    emissions_before_df = emissions_before_df.applymap(lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x)
    emissions_before_df = emissions_before_df.dropna(how='all')

    # 2. Compute total emissions without action per year
    emissions_without_action = emissions_before_df.sum(axis=0)
    emissions_without_action.index = emissions_without_action.index.str.extract(r'Emissions_(\d+)', expand=False).astype(int)
    emissions_by_year = emissions_without_action.sort_index()

    # DEBUG
    st.subheader("🔍 Annual emissions without actions")
    st.line_chart(emissions_by_year)

    # 3. Prepare reductions_by_solution_df
    reductions_by_solution_df = reductions_by_solution_df.copy()
    reductions_by_solution_df.columns = reductions_by_solution_df.columns.astype(int)

    # Debug
    st.subheader("🔍 Reductions by solution (raw input)")
    st.dataframe(reductions_by_solution_df)

    reductions_by_year = reductions_by_solution_df.T.sort_index()

    # Ensure same year index
    years = reductions_by_year.index
    emissions_by_year = emissions_by_year.loc[years]

    # 4. Compute trajectory = emissions - total_reduction
    total_reduction = reductions_by_year.sum(axis=1)
    trajectory = emissions_by_year - total_reduction

    # 5. Build bands: start from trajectory upwards
    bands = []
    bottom = trajectory.copy()
    for col in reductions_by_year.columns:
        top = bottom + reductions_by_year[col]
        bands.append((col, bottom.copy(), top.copy()))
        bottom = top

    # 6. Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each solution band
    for col, y_bottom, y_top in bands:
        color = solution_colors[col] if solution_colors and col in solution_colors else None
        ax.fill_between(years, y_bottom, y_top, label=col, color=color, alpha=0.8)

    # Curve: Emissions without action (black)
    ax.plot(emissions_by_year.index, emissions_by_year.values, color='black', linewidth=2, label='Emissions without action')

    # Curve: Trajectory (red)
    ax.plot(trajectory.index, trajectory.values, color='red', linewidth=2, label='Trajectory')


    # === Aesthetics ===

    # Title and labels
    ax.set_title("Annual CO₂e Emissions with and without Actions", fontsize=16, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Tonnes of CO₂e", fontsize=12)
    
    # Hide top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove grid
    ax.grid(False)
    
    # Background colors
    fig.patch.set_facecolor('white')        # Outside figure background
    ax.set_facecolor('#FAFAFA')             # Plot area background
    
    # Legend outside
    ax.legend(title="Solutions", loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, title_fontsize=12)
    
    # Fix y-axis starting at 0
    ax.set_ylim(bottom=0)
    if show_percentage_annotation:
        final_year = years.max()
        without = emissions_by_year.loc[final_year]
        with_action = trajectory.loc[final_year]
        percent_reduction = 100 * (1 - with_action / without)
        
        # Add reduction box below the legend area (right side)
        fig.text(0.88, 0.52,  # X and Y position in figure coordinates (tune as needed)
                 f"{percent_reduction:.0f}%\nreduction\nin {final_year}",
                 fontsize=12, color='red',
                 ha='center', va='top',
                 bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.4'))

    
        # Annotate final values
        ax.annotate(f"{int(without):,} tCO₂e", 
                    xy=(final_year, without),
                    xytext=(-40, 10),
                    textcoords="offset points",
                    ha='right',
                    fontsize=11,
                    color='black')
    
        ax.annotate(f"{int(with_action):,} tCO₂e", 
                    xy=(final_year, with_action),
                    xytext=(-40, -20),
                    textcoords="offset points",
                    ha='right',
                    fontsize=11,
                    color='red')
    
    # Adjust layout
    plt.tight_layout()


    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    st.download_button(
        label="📥 Download PNG",
        data=buffer.getvalue(),
        file_name="annual_emissions.png",
        mime="image/png"
    )

    return fig
