# =========================================================
# === VISUALISATION UTILITIES ===
# =========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import hashlib
import pandas as pd


# ----------------------------------------
# Helper for safe Streamlit keys
# ----------------------------------------
def _slug_key(text: str) -> str:
    """Generate a safe key for Streamlit widgets."""
    base = re.sub(r"[^a-zA-Z0-9\-]+", "-", str(text).strip().lower()).strip("-")
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
    return f"{base}-{h}"


# ----------------------------------------
# Color and order selection for solutions
# ----------------------------------------
def choose_solution_colors_and_order(solutions):
    """
    Allow user to pick colors and reorder mitigation solutions.
    Colors and order are stored in Streamlit session state.
    """
    st.subheader("🎨 Choose colors and order of solutions")

    if "solution_colors" not in st.session_state:
        st.session_state.solution_colors = {}
    if "solution_order" not in st.session_state:
        st.session_state.solution_order = list(solutions)

    # Synchronize order
    order = st.session_state.solution_order
    for s in solutions:
        if s not in order:
            order.append(s)
    order = [s for s in order if s in solutions]
    st.session_state.solution_order = order

    palette = plt.cm.tab20.colors
    def random_hex(): return "#" + "".join(np.random.choice(list("0123456789ABCDEF"), 6))

    # Reset buttons
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↺ Reset colors"):
            new_map = {s: "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                       for (s, (r, g, b)) in zip(order, palette)}
            st.session_state.solution_colors = new_map
            st.success("Colors reset.")
    with c2:
        if st.button("↻ Reset order (alphabetical)"):
            st.session_state.solution_order = sorted(order)
            st.rerun()

    # Editable list
    for i, s in enumerate(order):
        if s not in st.session_state.solution_colors:
            r, g, b = palette[i % len(palette)]
            st.session_state.solution_colors[s] = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

        cols = st.columns([3, 0.6, 0.6])
        with cols[0]:
            color = st.color_picker(
                f"{s}",
                value=st.session_state.solution_colors[s],
                key=f"solcolor_{_slug_key(s)}"
            )
            st.session_state.solution_colors[s] = color
        with cols[1]:
            if st.button("⬆️", key=f"solup_{i}") and i > 0:
                order[i-1], order[i] = order[i], order[i-1]
                st.session_state.solution_order = order
                st.rerun()
        with cols[2]:
            if st.button("⬇️", key=f"soldown_{i}") and i < len(order)-1:
                order[i+1], order[i] = order[i], order[i+1]
                st.session_state.solution_order = order
                st.rerun()


# ----------------------------------------
# Compute % share of total reduction
# ----------------------------------------
def compute_solution_percentages(impact_df, emissions_before_df):
    """
    Compute each solution's contribution *for the final year only*.

    - 'Total' = avoided emissions (tCO2e) in the final year (not a sum over years).
    - 'Share (%)' = 'Total' / (baseline emissions of the final year) * 100.
      This makes the sum of 'Share (%)' equal to the overall reduction % in that final year
      (e.g., 58%), not 100%.

    Parameters
    ----------
    impact_df : pd.DataFrame
        Index = solutions, Columns = years (e.g., 2025, 2026, ..., 2035) as int or 'YYYY' strings.
        Each cell contains avoided emissions (tCO2e) for that solution and year.
    emissions_before_df : pd.DataFrame
        Must contain columns named like 'Emissions_YYYY' (e.g., 'Emissions_2035').

    Returns
    -------
    pd.DataFrame
        Copy of impact_df with:
        - 'Total' (final-year avoided emissions per solution, tCO2e),
        - 'Share (%)' relative to baseline emissions in that final year,
        sorted by 'Total' descending.
    """
    import numpy as np

    df = impact_df.copy()

    # --- Identify final year column in impact_df (supports int or 'YYYY' string) ---
    # Collect candidate year columns
    year_cols = []
    for c in df.columns:
        if isinstance(c, (int, np.integer)):
            year_cols.append(int(c))
        elif isinstance(c, str) and c.isdigit() and len(c) == 4:
            year_cols.append(int(c))
    if not year_cols:
        raise ValueError("impact_df must have year columns (int or 'YYYY' strings).")

    final_year = max(year_cols)

    # Map back to the exact column name used in impact_df
    if final_year in df.columns:
        final_col = final_year
    else:
        final_col = str(final_year)

    # --- Take final-year avoided emissions as 'Total' (tCO2e) ---
    df["Total"] = df[final_col]

    # --- Get baseline (no-action) total emissions for the same final year ---
    emission_cols = [c for c in emissions_before_df.columns if c.startswith("Emissions_")]
    if not emission_cols:
        raise ValueError("No 'Emissions_YYYY' columns found in baseline data.")
    # Choose the corresponding final-year baseline column
    baseline_col = f"Emissions_{final_year}"
    if baseline_col not in emissions_before_df.columns:
        # Fallback: pick the latest available baseline year, but warn via caption
        # (keeps app running if data is slightly inconsistent)
        emission_cols_sorted = sorted(emission_cols, key=lambda x: int(x.split("_")[1]))
        baseline_col = emission_cols_sorted[-1]
        st.caption(
            f"⚠️ Using baseline '{baseline_col}' because 'Emissions_{final_year}' was not found."
        )
    baseline_final_total = emissions_before_df[baseline_col].sum()

    # Guard against divide-by-zero
    if baseline_final_total == 0:
        df["Share (%)"] = 0.0
    else:
        df["Share (%)"] = (df["Total"] / baseline_final_total * 100).round(2)

    # Optional: quick verification line (helps debug mismatches)
    total_reduction_pct = float(df["Share (%)"].sum())
    st.caption(
        f"📊 Final year used: **{final_year}** — total reduction vs baseline in {final_year}: **{total_reduction_pct}%**"
    )

    # Sort by final-year impact
    df = df.sort_values(by="Total", ascending=False)

    return df



# ----------------------------------------
# Generic SVG export
# ----------------------------------------
def export_svg(fig, filename="graph.svg"):
    """Export a Matplotlib figure to SVG for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue()

def plot_cumulative_emissions_reduction(
    emissions_before_df: pd.DataFrame,
    reductions_by_solution_df: pd.DataFrame,
    solution_colors: dict = None,
    show_percentage_annotation: bool = True
):
    """
    Plot cumulative CO2e emissions with:
    - Emissions without action (black line)
    - Emissions trajectory with actions (red line)
    - Stacked solution reductions between both
    """

    import matplotlib.pyplot as plt
    import io

    # 1️⃣ Clean baseline emissions
    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[
        [c for c in emissions_before_df.columns if c.startswith("Emissions_")]
    ].applymap(lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x).dropna(how="all")

    # 2️⃣ Compute baseline cumulative emissions
    emissions_without_action = emissions_before_df.sum(axis=0)
    emissions_without_action.index = emissions_without_action.index.str.extract(r"Emissions_(\d+)", expand=False).astype(int)
    emissions_cumulative = emissions_without_action.cumsum()

    # 3️⃣ Prepare reductions
    reductions_by_solution_df = reductions_by_solution_df.copy()
    # Convert only numeric columns to int
    reductions_by_solution_df.columns = [
        int(c) if str(c).isdigit() else c for c in reductions_by_solution_df.columns
    ]

    # Respect chosen order if defined
    if "solution_order" in st.session_state:
        reductions_by_solution_df = reductions_by_solution_df.loc[
            [s for s in st.session_state.solution_order if s in reductions_by_solution_df.index]
        ]

    # Legend labels with %
    if "Share (%)" in reductions_by_solution_df.columns:
        legend_labels = {
            name: f"{name} ({reductions_by_solution_df.loc[name, 'Share (%)']}%)"
            for name in reductions_by_solution_df.index
        }
    else:
        legend_labels = {name: name for name in reductions_by_solution_df.index}

    reductions_by_year = reductions_by_solution_df[
        [c for c in reductions_by_solution_df.columns if isinstance(c, int)]
    ].T
    reductions_cumulative = reductions_by_year.cumsum()

    # 4️⃣ Common year range (filter only numeric years)
    years_emissions = emissions_cumulative.index
    years_reductions = reductions_cumulative.index
    years = sorted([int(y) for y in set(years_emissions).union(years_reductions) if str(y).isdigit()])

    reductions_cumulative = reductions_cumulative.reindex(years, method="ffill").fillna(0)
    emissions_cumulative = emissions_cumulative.reindex(years, method="ffill").fillna(0)

    # 5️⃣ Compute trajectory
    total_reduction = reductions_cumulative.sum(axis=1)
    trajectory = emissions_cumulative - total_reduction

    # 6️⃣ Build stacked bands
    bands = []
    bottom = trajectory.copy()
    for col in reductions_cumulative.columns:
        top = bottom + reductions_cumulative[col]
        bands.append((col, bottom.copy(), top.copy()))
        bottom = top

    # 7️⃣ Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    for col, y_bottom, y_top in bands:
        color = solution_colors[col] if solution_colors and col in solution_colors else None
        ax.fill_between(years, y_bottom, y_top, label=legend_labels.get(col, col), color=color, alpha=0.8)

    ax.plot(emissions_cumulative.index, emissions_cumulative.values, color="black", linewidth=2, label="Emissions without action")
    ax.plot(trajectory.index, trajectory.values, color="red", linewidth=2, label="Trajectory")

    # Style
    ax.set_title("Cumulative CO₂e Emissions with and without Actions", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tonnes of CO₂e")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.legend(title="Solutions", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, title_fontsize=12)

    # Percentage annotation
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

    # Download
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
    Plot annual CO₂e emissions with and without actions.
    """

    import matplotlib.pyplot as plt
    import io

    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[
        [c for c in emissions_before_df.columns if c.startswith("Emissions_")]
    ].applymap(lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x).dropna(how="all")

    emissions_without_action = emissions_before_df.sum(axis=0)
    emissions_without_action.index = emissions_without_action.index.str.extract(r"Emissions_(\d+)", expand=False).astype(int)
    emissions_by_year = emissions_without_action.sort_index()

    reductions_by_solution_df = reductions_by_solution_df.copy()
    reductions_by_solution_df.columns = [
        int(c) if str(c).isdigit() else c for c in reductions_by_solution_df.columns
    ]

    if "solution_order" in st.session_state:
        reductions_by_solution_df = reductions_by_solution_df.loc[
            [s for s in st.session_state.solution_order if s in reductions_by_solution_df.index]
        ]

    if "Share (%)" in reductions_by_solution_df.columns:
        legend_labels = {
            name: f"{name} ({reductions_by_solution_df.loc[name, 'Share (%)']}%)"
            for name in reductions_by_solution_df.index
        }
    else:
        legend_labels = {name: name for name in reductions_by_solution_df.index}

    reductions_by_year = reductions_by_solution_df[
        [c for c in reductions_by_solution_df.columns if isinstance(c, int)]
    ].T.sort_index()

    years = sorted([int(y) for y in set(emissions_by_year.index).union(reductions_by_year.index) if str(y).isdigit()])
    reductions_by_year = reductions_by_year.reindex(years, method="ffill").fillna(0)
    emissions_by_year = emissions_by_year.reindex(years, method="ffill").fillna(0)

    total_reduction = reductions_by_year.sum(axis=1)
    trajectory = emissions_by_year - total_reduction

    bands = []
    bottom = trajectory.copy()
    for col in reductions_by_year.columns:
        top = bottom + reductions_by_year[col]
        bands.append((col, bottom.copy(), top.copy()))
        bottom = top

    fig, ax = plt.subplots(figsize=(14, 8))
    for col, y_bottom, y_top in bands:
        color = solution_colors[col] if solution_colors and col in solution_colors else None
        ax.fill_between(years, y_bottom, y_top, label=legend_labels.get(col, col), color=color, alpha=0.8)

    ax.plot(emissions_by_year.index, emissions_by_year.values, color="black", linewidth=2, label="Emissions without action")
    ax.plot(trajectory.index, trajectory.values, color="red", linewidth=2, label="Trajectory")

    ax.set_title("Annual CO₂e Emissions with and without Actions", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tonnes of CO₂e")
    ax.set_ylim(bottom=0)
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Solutions", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, title_fontsize=12)

    if show_percentage_annotation:
        final_year = years[-1]
        start_year = years[0]
        without = emissions_by_year.loc[final_year]
        without1 = emissions_by_year.loc[start_year]
        with_action = trajectory.loc[final_year]
        relative_percent_reduction = 100 * (1 - with_action / without)
        absolute_percent_reduction = 100 * (1 - with_action / without1)
        fig.text(
            0.88, 0.52,
            f"{relative_percent_reduction:.0f}%\nrelative_percenatge_reduction\nin {final_year}",
            fontsize=12, color="red",
            ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.4")
        )
        fig.text(
            0.88, 0.22,
            f"{absolute_percent_reduction:.0f}%\nabsolute_percentage_reduction\nin {final_year}",
            fontsize=12, color="red",
            ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.4")
        )

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
    Prepare emissions and reductions data for waterfall chart.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    available_years = [int(c.split("_")[1]) for c in emissions_before_df.columns if c.startswith("Emissions_")]
    data_start, data_end = (min(available_years), max(available_years)) if available_years else (2025, 2035)

    if "years" in st.session_state and st.session_state["years"]:
        start_year = min(data_start, min(st.session_state["years"]))
        end_year = max(data_end, max(st.session_state["years"]))
    else:
        start_year, end_year = data_start, data_end

    emissions_before_df = emissions_before_df.copy()
    emissions_before_df = emissions_before_df[[c for c in emissions_before_df.columns if c.startswith("Emissions_")]]
    emissions_before_df = emissions_before_df.applymap(lambda x: float(str(x).replace(",", "")) if pd.notnull(x) else x).dropna(how="all")

    col_start, col_target = f"Emissions_{start_year}", f"Emissions_{end_year}"
    start_value = emissions_before_df[col_start].sum() if col_start in emissions_before_df.columns else 0
    no_action_value = emissions_before_df[col_target].sum() if col_target in emissions_before_df.columns else 0

    reductions_by_solution_df = reductions_by_solution_df.copy()
    reductions_by_solution_df.columns = [
        int(c) if str(c).isdigit() else c for c in reductions_by_solution_df.columns
    ]

    if "solution_order" in st.session_state:
        reductions_by_solution_df = reductions_by_solution_df.loc[
            [s for s in st.session_state.solution_order if s in reductions_by_solution_df.index]
        ]

    reductions_in_year = (
        reductions_by_solution_df[end_year]
        if end_year in reductions_by_solution_df.columns
        else pd.Series(0, index=reductions_by_solution_df.index)
    )

    steps = [no_action_value]
    current_value = no_action_value
    for reduction in reductions_in_year:
        current_value -= reduction
        steps.append(current_value)

    if "Share (%)" in reductions_by_solution_df.columns:
        labels = [f"{end_year} emissions (no action)"] + [
            f"{name} ({reductions_by_solution_df.loc[name, 'Share (%)']}%)"
            for name in reductions_in_year.index
        ]
    else:
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

    Improvements:
    -------------
    - Automatically respects solution order (already handled upstream)
    - Displays solution shares (e.g. "Bike plan (12%)") if available in labels
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    import numpy as np
    import textwrap
    import io

    def wrap_label(label: str, max_chars: int = 15) -> str:
        """
        Automatically wraps a label into multiple lines based on max character length.
        Splits on word boundaries.
        """
        return "\n".join(textwrap.wrap(label, width=max_chars))

    # === 1️⃣ Setup main bars ===
    n = len(steps) + 1  # total bars = start + steps
    x_main = np.arange(n)
    y_main = [start_value] + steps  # emission levels after each step

    # === 2️⃣ Create figure ===
    fig, ax = plt.subplots(figsize=(14, 6))

    # === 3️⃣ Main bars ===
    for i in range(n):
        bar_color = colors[i] if i < len(colors) else colors[-1]
        ax.bar(x_main[i], y_main[i], color=bar_color, width=0.6)
        ax.text(x_main[i], y_main[i] + max(y_main) * 0.02, f"{int(y_main[i]):,} t", ha="center", fontsize=11)

    # === 4️⃣ Intermediate delta bars and arrow labels ===
    for i in range(1, n):
        ymin = min(y_main[i - 1], y_main[i])
        delta = abs(y_main[i] - y_main[i - 1])
        x_middle = (x_main[i - 1] + x_main[i]) / 2

        next_color = colors[i] if i < len(colors) else "lightgrey"
        transition_color = to_rgba(next_color, alpha=0.3)

        ax.bar(
            x_middle, delta, bottom=ymin,
            width=0.3, color=transition_color, edgecolor="none", zorder=2
        )

        # Label + arrow
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

    # === 5️⃣ Axes and layout ===
    wrapped_labels = [wrap_label(label, max_chars=wrap_char_limit) for label in labels]
    full_labels = ["Start"] + wrapped_labels

    ax.set_xticks(x_main)
    ax.set_xticklabels(full_labels, fontsize=11)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Tonnes of CO₂e")

    ymax = max(y_main) * 1.3
    ax.set_ylim(0, ymax)

    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # === 6️⃣ Export PNG ===
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    st.download_button(
        label="📥 Download PNG",
        data=buffer.getvalue(),
        file_name="waterfall_emissions.png",
        mime="image/png"
    )

    return fig
