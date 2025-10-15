# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:29:57 2025

@author: quent
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import textwrap
from matplotlib.colors import to_rgba


def choose_solution_colors(solutions):
    """
    Display a color picker for each solution and store selections in session state.

    This function allows users to assign a custom color to each solution using Streamlit's
    color picker widget. If a color has not yet been set, a random color is generated.

    Parameters:
    - solutions (List[str]): List of solution names for which colors must be selected.

    Effects:
    - Updates st.session_state["solution_colors"] with the selected color for each solution.
    """
    st.subheader("🎨 Choose a color for each solution")

    if "solution_colors" not in st.session_state:
        st.session_state.solution_colors = {}

    for sol in solutions:
        if sol not in st.session_state.solution_colors:
            random_color = "#" + ''.join(np.random.choice(list("0123456789ABCDEF"), 6))
            st.session_state.solution_colors[sol] = random_color

        color = st.color_picker(
            f"Color for {sol}",
            value=st.session_state.solution_colors[sol],
            key=f"color_picker_solution_{sol}"
        )

        st.session_state.solution_colors[sol] = color

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
    """

    # 1. Clean and sanitize input
    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[
        [col for col in emissions_before_df.columns if col.startswith("Emissions_")]
    ]
    emissions_before_df = emissions_before_df.applymap(
        lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x
    )
    emissions_before_df = emissions_before_df.dropna(how="all")

    # 2. Compute emissions without action (baseline)
    emissions_without_action = emissions_before_df.sum(axis=0)
    emissions_without_action.index = (
        emissions_without_action.index.str.extract(r"Emissions_(\d+)", expand=False).astype(int)
    )
    emissions_cumulative = emissions_without_action.cumsum()

    # 3. Prepare reductions_by_solution_df
    reductions_by_solution_df = reductions_by_solution_df.copy()
    reductions_by_solution_df.columns = reductions_by_solution_df.columns.astype(int)
    reductions_by_year = reductions_by_solution_df.T
    reductions_cumulative = reductions_by_year.cumsum()

    # 4. Ensure both datasets share the same full year range
    years_emissions = emissions_cumulative.index
    years_reductions = reductions_cumulative.index
    years = sorted(set(years_emissions).union(years_reductions))

    reductions_cumulative = reductions_cumulative.reindex(years, method="ffill").fillna(0)
    emissions_cumulative = emissions_cumulative.reindex(years, method="ffill").fillna(0)

    # 5. Compute trajectory
    total_reduction = reductions_cumulative.sum(axis=1)
    trajectory = emissions_cumulative - total_reduction

    # 6. Build stacked bands
    bands = []
    bottom = trajectory.copy()
    for col in reductions_cumulative.columns:
        top = bottom + reductions_cumulative[col]
        bands.append((col, bottom.copy(), top.copy()))
        bottom = top

    # 7. Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    for col, y_bottom, y_top in bands:
        color = solution_colors[col] if solution_colors and col in solution_colors else None
        ax.fill_between(years, y_bottom, y_top, label=col, color=color, alpha=0.8)

    ax.plot(emissions_cumulative.index, emissions_cumulative.values, color="black", linewidth=2, label="Emissions without action")
    ax.plot(trajectory.index, trajectory.values, color="red", linewidth=2, label="Trajectory")

    # === Style ===
    ax.set_title("Cumulative CO₂e Emissions with and without Actions", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tonnes of CO₂e")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.legend(title="Solutions", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, title_fontsize=12)

    # === Reduction box ===
    if show_percentage_annotation:
        final_year = years[-1]
        without = emissions_cumulative.loc[final_year]
        with_action = trajectory.loc[final_year]
        percent_reduction = 100 * (1 - with_action / without)

        fig.text(
            0.88, 0.52,
            f"{percent_reduction:.0f}%\nreduction\nin {final_year}",
            fontsize=12, color="red",
            ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.4")
        )

    # === Download ===
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    st.download_button("📥 Download PNG", data=buffer.getvalue(), file_name="cumulative_emissions.png", mime="image/png")

    return fig





def plot_annual_emissions_reduction(
    emissions_before_df: pd.DataFrame,
    reductions_by_solution_df: pd.DataFrame,
    solution_colors: dict = None,
    show_percentage_annotation: bool = True
):
    """
    Plot annual CO₂e emissions with and without actions (non-cumulative).
    """

    # 1. Clean and prepare data
    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[
        [col for col in emissions_before_df.columns if col.startswith("Emissions_")]
    ]
    emissions_before_df = emissions_before_df.applymap(
        lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x
    )
    emissions_before_df = emissions_before_df.dropna(how="all")

    # 2. Compute baseline
    emissions_without_action = emissions_before_df.sum(axis=0)
    emissions_without_action.index = (
        emissions_without_action.index.str.extract(r"Emissions_(\d+)", expand=False).astype(int)
    )
    emissions_by_year = emissions_without_action.sort_index()

    # 3. Prepare reductions
    reductions_by_solution_df = reductions_by_solution_df.copy()
    reductions_by_solution_df.columns = reductions_by_solution_df.columns.astype(int)
    reductions_by_year = reductions_by_solution_df.T.sort_index()

    # 4. Ensure same full year range
    years_emissions = emissions_by_year.index
    years_reductions = reductions_by_year.index
    years = sorted(set(years_emissions).union(years_reductions))

    reductions_by_year = reductions_by_year.reindex(years, method="ffill").fillna(0)
    emissions_by_year = emissions_by_year.reindex(years, method="ffill").fillna(0)

    # 5. Compute trajectory
    total_reduction = reductions_by_year.sum(axis=1)
    trajectory = emissions_by_year - total_reduction

    # 6. Build bands
    bands = []
    bottom = trajectory.copy()
    for col in reductions_by_year.columns:
        top = bottom + reductions_by_year[col]
        bands.append((col, bottom.copy(), top.copy()))
        bottom = top

    # 7. Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    for col, y_bottom, y_top in bands:
        color = solution_colors[col] if solution_colors and col in solution_colors else None
        ax.fill_between(years, y_bottom, y_top, label=col, color=color, alpha=0.8)

    ax.plot(emissions_by_year.index, emissions_by_year.values, color="black", linewidth=2, label="Emissions without action")
    ax.plot(trajectory.index, trajectory.values, color="red", linewidth=2, label="Trajectory")

    # === Style ===
    ax.set_title("Annual CO₂e Emissions with and without Actions", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tonnes of CO₂e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.legend(title="Solutions", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, title_fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(False)

    # === Reduction annotation ===
    if show_percentage_annotation:
        final_year = years[-1]
        without = emissions_by_year.loc[final_year]
        with_action = trajectory.loc[final_year]
        percent_reduction = 100 * (1 - with_action / without)

        fig.text(
            0.88, 0.52,
            f"{percent_reduction:.0f}%\nreduction\nin {final_year}",
            fontsize=12, color="red",
            ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.4")
        )

    # === Download ===
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    st.download_button("📥 Download PNG", data=buffer.getvalue(), file_name="annual_emissions.png", mime="image/png")

    return fig



def prepare_waterfall_inputs(
    emissions_before_df: pd.DataFrame,
    reductions_by_solution_df: pd.DataFrame,
    solution_colors: dict = None
):
    """
    Prepares emissions and reduction data for the waterfall chart.
    """

    import matplotlib.pyplot as plt

    # 1. Get actual data years
    available_years = [
        int(c.split("_")[1]) for c in emissions_before_df.columns if c.startswith("Emissions_")
    ]
    if available_years:
        data_start = min(available_years)
        data_end = max(available_years)
    else:
        data_start, data_end = 2025, 2035

    # 2. Combine with session info if available
    if "years" in st.session_state and st.session_state["years"]:
        start_year = min(data_start, min(st.session_state["years"]))
        end_year = max(data_end, max(st.session_state["years"]))
    else:
        start_year, end_year = data_start, data_end

    # 3. Sanitize data
    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[
        [col for col in emissions_before_df.columns if col.startswith("Emissions_")]
    ]
    emissions_before_df = emissions_before_df.applymap(
        lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x
    ).dropna(how="all")

    col_start = f"Emissions_{start_year}"
    col_target = f"Emissions_{end_year}"

    start_value = emissions_before_df[col_start].sum() if col_start in emissions_before_df.columns else 0
    no_action_value = emissions_before_df[col_target].sum() if col_target in emissions_before_df.columns else 0

    # 4. Cumulative reductions for end year
    reductions_by_solution_df = reductions_by_solution_df.copy()
    reductions_by_solution_df.columns = reductions_by_solution_df.columns.astype(int)
    reductions_in_year = (
        reductions_by_solution_df[end_year]
        if end_year in reductions_by_solution_df.columns
        else pd.Series(0, index=reductions_by_solution_df.index)
    )

    # 5. Apply reductions one by one
    steps = [no_action_value]
    current_value = no_action_value
    for reduction in reductions_in_year:
        current_value -= reduction
        steps.append(current_value)

    # 6. Labels and colors
    labels = [f"{end_year} emissions (no action)"] + [str(name) for name in reductions_in_year.index]
    default_colors = plt.cm.tab20.colors
    colors = ["#ED6D2D", "#ED6D2D"]
    for i, name in enumerate(reductions_in_year.index):
        if solution_colors and name in solution_colors:
            colors.append(solution_colors[name])
        else:
            colors.append(default_colors[i % len(default_colors)])

    return start_value, steps, labels, colors


def plot_waterfall_emissions(
    start_value: float,
    steps: list,
    labels: list,
    colors: list,
    intermediate_color: str = "#B0C4DE",
    title: str = "Emissions Waterfall Chart",
    wrap_char_limit: int = 15
):
    """
    Plot a waterfall CO2e emissions chart with:
    - A start value (e.g. emissions in start year)
    - Emissions after each successive solution
    - Bars showing each step, with lighter intermediate bars for deltas
    - Arrows with wrapped labels indicating the cause of each change
    - Annotated values on each main bar

    Parameters:
    - start_value: float, initial emissions (e.g. 2022 level)
    - steps: list of floats, emission levels after each step (starting with last year without actions)
    - labels: list of str, labels for each step (should match length of steps)
    - colors: list of str, main bar colors (should match len(steps) + 1 with start)
    - intermediate_color: str, color of lighter delta bars
    - title: str, title of the plot
    - wrap_char_limit: int, max number of characters per line in labels

    Returns:
    - fig (matplotlib.figure.Figure)
    """

    def wrap_label(label: str, max_chars: int = 15) -> str:
        """
        Automatically wraps a label into multiple lines based on max character length.
        Splits on word boundaries.
        """
        return "\n".join(textwrap.wrap(label, width=max_chars))

    # === 1. Setup bar positions and values ===
    n = len(steps) + 1  # total bars = start + steps
    x_main = np.arange(n)
    y_main = [start_value] + steps  # emissions levels after each step

    # === 2. Create figure and axis ===
    fig, ax = plt.subplots(figsize=(14, 6))

    # === 3. Plot main bars ===
    for i in range(n):
        bar_color = colors[i] if i < len(colors) else colors[-1]
        ax.bar(x_main[i], y_main[i], color=bar_color, width=0.6)
        ax.text(x_main[i], y_main[i] + max(y_main) * 0.02, f"{int(y_main[i]):,} t", ha="center", fontsize=11)


    # === 4. Plot intermediate (delta) bars and annotate ===
    for i in range(1, n):
        ymin = min(y_main[i - 1], y_main[i])
        delta = abs(y_main[i] - y_main[i - 1])
        x_middle = (x_main[i - 1] + x_main[i]) / 2
    
        # Get color of the next main bar and convert to transparent (alpha)
        next_color = colors[i] if i < len(colors) else "lightgrey"
        transition_color = to_rgba(next_color, alpha=0.3)
    
        # Delta bar (transition bar)
        ax.bar(
            x_middle, delta, bottom=ymin,
            width=0.3, color=transition_color, edgecolor="none", zorder=2
        )
    
        # Label for arrow annotation
        label = labels[i - 1] if i - 1 < len(labels) else f"Step {i}"
        wrapped_arrow_label = wrap_label(label, max_chars=wrap_char_limit)
        arrow_y = ymin + delta + max(y_main) * 0.01
        arrow_offset = max(y_main) * 0.05
    
        ax.annotate(
            wrapped_arrow_label,
            xy=(x_middle, arrow_y),
            xytext=(x_middle, arrow_y + arrow_offset),
            ha="center",
            fontsize=11,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2")
        )

    # === 5. Axis styling ===
    # Labels & title
    wrapped_labels = [wrap_label(label, max_chars=wrap_char_limit) for label in labels]
    full_labels = ["Start"] + wrapped_labels
    
    ax.set_xticks(x_main)
    ax.set_xticklabels(full_labels, fontsize=11)
    
    # Title with space
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    # Axes
    ax.set_ylabel("Tonnes of CO₂e")
    
    # Leave room on top for arrows
    ymax = max(y_main) * 1.3
    ax.set_ylim(0, ymax)
    
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # === 6. Export as PNG in Streamlit ===
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    st.download_button(
        label="📥 Download PNG",
        data=buffer.getvalue(),
        file_name="waterfall_emissions.png",
        mime="image/png"
    )

    return fig
