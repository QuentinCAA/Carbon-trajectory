# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:38:30 2025

@author: quent
"""
import streamlit as st
from streamlit_tree_select import tree_select
from modules.tree import build_tree
import pandas as pd


def define_inflation(years):
    """
    Allow the user to define inflation rates per year, either manually or from preset country data.

    This function provides two options:
    1. Use preset IMF-based inflation projections (via TheGlobalEconomy.com)
       for France, Switzerland, the EU, and the USA, extended to 2035.
    2. Enter custom inflation rates manually.

    Parameters:
    - years (List[int]): List of target years (e.g. [2025, 2026, ..., 2035]).

    Effects:
    - Updates st.session_state["inflation_rates"] as a dictionary {year: inflation_rate}.
    - Updates st.session_state["inflation_source"] with the source name for transparency.
    """

    st.subheader("Inflation settings (annual %)")

    # --- 1. Preset inflation data (IMF WEO via TheGlobalEconomy.com, extended to 2035) ---
    presets = {
        "France": {
            "rates": {
                2025: 1.67, 2026: 1.25, 2027: 2.16, 2028: 1.86, 2029: 1.96, 2030: 1.72,
                2031: 1.72, 2032: 1.72, 2033: 1.72, 2034: 1.72, 2035: 1.72
            },
            "source": "IMF World Economic Outlook (via TheGlobalEconomy.com)"
        },
        "Switzerland": {
            "rates": {
                2025: 0.24, 2026: 0.53, 2027: 0.72, 2028: 0.70, 2029: 0.72, 2030: 0.70,
                2031: 0.70, 2032: 0.70, 2033: 0.70, 2034: 0.70, 2035: 0.70
            },
            "source": "IMF World Economic Outlook (via TheGlobalEconomy.com)"
        },
        "European Union": {
            "rates": {
                2025: 2.20, 2026: 2.00, 2027: 1.90, 2028: 2.00, 2029: 2.00, 2030: 1.90,
                2031: 1.90, 2032: 1.90, 2033: 1.90, 2034: 1.90, 2035: 1.90
            },
            "source": "IMF World Economic Outlook – Euro area projection (via TheGlobalEconomy.com)"
        },
        "USA": {
            "rates": {
                2025: 2.88, 2026: 2.27, 2027: 2.37, 2028: 2.17, 2029: 2.18, 2030: 2.18,
                2031: 2.18, 2032: 2.18, 2033: 2.18, 2034: 2.18, 2035: 2.18
            },
            "source": "IMF World Economic Outlook (via TheGlobalEconomy.com)"
        },
        "Global average": {
            "rates": {
                2025: 3.00, 2026: 2.90, 2027: 2.80, 2028: 2.70, 2029: 2.70, 2030: 2.60,
                2031: 2.60, 2032: 2.60, 2033: 2.60, 2034: 2.60, 2035: 2.60
            },
            "source": "IMF World Economic Outlook (global inflation average)"
        }
    }

    # --- 2. Initialize session variables ---
    if "inflation_rates" not in st.session_state:
        st.session_state.inflation_rates = {y: 0.0 for y in years}
    if "inflation_source" not in st.session_state:
        st.session_state.inflation_source = "Manual input"

    # --- 3. Let the user choose between manual or preset mode ---
    mode = st.radio(
        "How do you want to define inflation?",
        ["Manual input", "Use preset values"],
        horizontal=True
    )

    # --- 4. Preset mode: user selects a country or region ---
    if mode == "Use preset values":
        country = st.selectbox("Select a preset dataset:", list(presets.keys()))
        preset = presets[country]
        base_rates = preset["rates"]
        source = preset["source"]

        # Match selected years and fill in any missing ones (if years differ)
        inflation_values = {}
        max_known_year = max(base_rates.keys())
        last_known_rate = base_rates[max_known_year]

        for y in years:
            if y in base_rates:
                inflation_values[y] = base_rates[y]
            elif y > max_known_year:
                inflation_values[y] = last_known_rate  # extend stable value
            else:
                # For years before available data, use closest known rate
                closest = min(base_rates.keys(), key=lambda k: abs(k - y))
                inflation_values[y] = base_rates[closest]

        st.session_state.inflation_rates = inflation_values
        st.session_state.inflation_source = f"Preset: {country} ({source})"

        st.success(f"✅ Inflation preset '{country}' loaded.")

        # Display as a dataframe for transparency
        df = pd.DataFrame.from_dict(inflation_values, orient="index", columns=["Inflation (%)"])
        df.index.name = "Year"
        st.dataframe(df)

    # --- 5. Manual mode: user enters each year's inflation ---
    else:
        with st.form("manual_inflation_form"):
            for y in years:
                key = f"infl_{y}"
                default = st.session_state.inflation_rates.get(y, 0.0)
                st.session_state.inflation_rates[y] = st.number_input(
                    f"Inflation rate for {y} (%)",
                    min_value=-5.0, max_value=20.0, step=0.1,
                    value=default, format="%.2f", key=key
                )
            submitted = st.form_submit_button("Save custom inflation")
            if submitted:
                st.session_state.inflation_source = "Manual input"
                st.success("✅ Custom inflation rates saved.")

    # --- 6. Transparency and data source notice ---
    st.markdown(
        """
        <div style='color:grey; font-size:0.9em;'>
        ℹ️ <strong>Data source:</strong> IMF World Economic Outlook (via 
        <a href='https://www.theglobaleconomy.com/rankings/inflation_outlook_imf/' target='_blank'>
        TheGlobalEconomy.com</a>).<br>
        Values beyond 2030 are extrapolated assuming stable long-term inflation for transparency and continuity.
        </div>
        """,
        unsafe_allow_html=True
    )


def create_growth(years):
    """
    Create a new growth-based or budget-based projection and store it in session state.
    
    This function displays a form where the user can define a projection by either
    specifying a fixed growth percentage or a budget evolution over time. The form
    supports optional intermediate budget points and stores the result in session state.
    
    Parameters:
    - years (List[int]): List of target years, including the start and end years for the projection.
    
    Effects:
    - Updates st.session_state["growth_inputs"] with a new projection object containing:
        - 'name': The label of the projection.
        - 'mode': Either "Growth %" or "Budget Projection".
        - 'growth': Growth percentage (if applicable).
        - 'budget': Dictionary of budgets per year (if applicable).
        - 'categories': Placeholder for future tree-based category assignment.
    - Displays success confirmation in the Streamlit interface.
    """
    
    st.subheader("Create a new growth or budget projection")
    
    # 🛈 Helper note for users
    st.markdown(
        """
        <div style='color: grey; font-size: 0.9em;'>
        💡 To create a <strong>Budget Projection</strong>, simply select <strong>Budget Projection</strong> 
        and click <strong>Add projection</strong> without filling any intermediate values.<br>
        This will initialize an empty projection that you can edit later.<br><br>
        """,
        unsafe_allow_html=True
    )

    if "growth_inputs" not in st.session_state:
        st.session_state.growth_inputs = []

    start_year = years[0]
    end_year = years[-1]
    intermediate_years = [y for y in years if y != start_year and y != end_year]

    with st.form("create_growth_form"):
        name = st.text_input("Name of the projection (e.g. 'General org', 'Events')")
        mode = st.radio("Type of projection", ["Growth %", "Budget Projection"])

        growth = None
        budget = {}

        if mode == "Growth %":
            growth = st.number_input(
                "Growth percentage (%) — e.g. -3 for -3%/year, +4 for +4%/year",
                min_value=-100.0,
                max_value=100.0,
                format="%.2f"
            )
        else:
            budget_start = st.number_input(f"Budget in {start_year} (€)", min_value=0.0, format="%.2f")
            budget_end = st.number_input(f"Budget in {end_year} (€)", min_value=0.0, format="%.2f")

            selected_intermediates = st.multiselect(
                "Add optional intermediate years",
                intermediate_years
            )

            for y in selected_intermediates:
                val = st.number_input(
                    f"Budget in {y} (€)",
                    min_value=0.0,
                    format="%.2f",
                    key=f"budget_{name}_{y}"
                )
                budget[y] = val

            budget[str(start_year)] = budget_start
            budget[str(end_year)] = budget_end

        submitted = st.form_submit_button("Add projection")

        if submitted and name:
            new_proj = {
                "name": name,
                "mode": mode,
                "growth": growth,
                "budget": budget,
                "categories": {}  # placeholder for tree selection
            }

            st.session_state.growth_inputs.append(new_proj)
            st.success(f"✅ Projection '{name}' added.")



def assign_growth(data):
    """
    Allow the user to assign categories to existing growth or budget projections,
    and delete projections if needed.
    
    Parameters:
    - data (pd.DataFrame): Emissions data used to construct the hierarchical tree structure.
    
    Effects:
    - For each projection in st.session_state["growth_inputs"], displays a form with:
        - Projection name and type (Growth % or Budget).
        - Existing growth or budget values.
        - A tree selector for assigning categories.
        - A delete button to remove the projection.
    - Updates the 'categories' field of each projection based on user selections.
    """
    st.subheader("Assign growth or budget projections to categories")
    
    # 🛈 Helper note for users
    st.markdown(
        """
        <div style='color: grey; font-size: 0.9em;'>
        You can <strong>delete a projection</strong> later if it's no longer needed.<br>
        To do so, first click <strong>🗑️ Delete</strong> — this will remove its assigned effects — 
        then click <strong>Save configuration</strong> to confirm and remove it permanently.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "growth_inputs" not in st.session_state or not st.session_state.growth_inputs:
        st.info("No projections available.")
        return

    tree = build_tree(data)
    col1, col2 = st.columns(2)
    to_delete = []

    for i, g in enumerate(st.session_state.growth_inputs):
        col = col1 if i % 2 == 0 else col2

        with col.form(f"assign_growth_{i}"):
            st.markdown(f"### 🛠️ {g['name']} ({g['mode']})")

            if g["mode"] == "Growth %":
                st.write(f"Growth rate: **{g['growth']:+.2f}% per year**")
            else:
                st.write("Budget projection:")
                for year, amount in sorted(g["budget"].items(), key=lambda x: int(x[0])):
                    st.markdown(f"- **{year}**: {amount:,.2f} €")

            checked = g["categories"].get("checked", []) if isinstance(g["categories"], dict) else []
            selection = tree_select(tree, checked=checked, key=f"growth_tree_{i}")
            g["categories"] = selection

            col_save, col_del = st.columns([3, 1])
            with col_save:
                submitted = st.form_submit_button("Save configuration")
            with col_del:
                delete_clicked = st.form_submit_button("🗑️ Delete", type="secondary")

            if submitted:
                st.success(f"✅ Categories updated for '{g['name']}'.")
            if delete_clicked:
                to_delete.append(i)

    # Remove deleted projections after iteration
    if to_delete:
        for idx in sorted(to_delete, reverse=True):
            deleted_name = st.session_state.growth_inputs[idx]["name"]
            del st.session_state.growth_inputs[idx]
            st.warning(f"🗑️ Projection '{deleted_name}' deleted.")



def summarize_growths(years):
    """
    Summarize nominal and real growth rates for all defined projections.

    This function compares the nominal (user-entered or derived from budgets)
    and inflation-adjusted (real) growth rates for each projection defined
    in st.session_state["growth_inputs"].

    It uses the inflation data defined via define_inflation() to compute real growth.

    Parameters:
    - years (List[int]): List of target years (e.g. [2025, 2026, ..., 2035]).

    Returns:
    - pd.DataFrame: Summary with one row per projection showing nominal, average inflation, and real growth (%).
    """

    import pandas as pd

    if "growth_inputs" not in st.session_state or not st.session_state.growth_inputs:
        st.info("No growth or budget projections defined.")
        return None

    inflation_rates = st.session_state.get("inflation_rates", {})
    inflation_source = st.session_state.get("inflation_source", "Not specified")

    summary_data = []

    for g in st.session_state.growth_inputs:
        mode = g["mode"]
        name = g["name"]

        # === Case 1: Growth % projections ===
        if mode == "Growth %":
            nominal = g["growth"]
            inflation_values = [inflation_rates.get(y, 0) for y in years[1:]]
            avg_inflation = sum(inflation_values) / len(inflation_values) if inflation_values else 0
            real = ((1 + nominal / 100) / (1 + avg_inflation / 100) - 1) * 100

            summary_data.append({
                "Projection": name,
                "Type": "Growth %",
                "Nominal growth (%)": round(nominal, 2),
                "Average inflation (%)": round(avg_inflation, 2),
                "Real growth (%)": round(real, 2)
            })

        # === Case 2: Budget Projections ===
        elif mode == "Budget Projection":
            budget = g.get("budget", {})
            valid_years = sorted([int(y) for y, v in budget.items() if v > 0])

            if len(valid_years) >= 2:
                y0, y_end = valid_years[0], valid_years[-1]
                b0, b_end = budget[str(y0)], budget[str(y_end)]

                # Nominal CAGR (Compound Annual Growth Rate)
                nominal_cagr = ((b_end / b0) ** (1 / (y_end - y0)) - 1) * 100

                # Average inflation for that period only
                inflation_values = [inflation_rates.get(y, 0) for y in range(y0 + 1, y_end + 1)]
                avg_inflation = sum(inflation_values) / len(inflation_values) if inflation_values else 0

                # Inflation-adjusted (real) CAGR
                inflation_factor = 1.0
                for yr in range(y0 + 1, y_end + 1):
                    inflation_factor *= (1 + inflation_rates.get(yr, 0) / 100)

                real_cagr = (((b_end / (b0 * inflation_factor)) ** (1 / (y_end - y0))) - 1) * 100
            else:
                nominal_cagr = 0
                avg_inflation = 0
                real_cagr = 0

            summary_data.append({
                "Projection": name,
                "Type": "Budget Projection",
                "Nominal growth (%)": round(nominal_cagr, 2),
                "Average inflation (%)": round(avg_inflation, 2),
                "Real growth (%)": round(real_cagr, 2)
            })

    # === Display the summary ===
    st.markdown("### 📊 Growth summary (nominal vs. real)")
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)

    st.markdown(
        f"<div style='color:grey; font-size:0.9em;'>"
        f"ℹ️ Real growth is computed using inflation data from: "
        f"<strong>{inflation_source}</strong>.<br>"
        f"Nominal growth represents raw input values; real growth adjusts for average inflation during the same period."
        f"</div>",
        unsafe_allow_html=True
    )

    return df_summary


def apply_projections_to_base(projection_df, years):
    """
    Apply growth or budget projections to the projection DataFrame based on user-defined inputs,
    adjusted for inflation when available.

    In addition to applying projections, this version computes inflation-adjusted (real) factors.
    (Display of summaries moved out; this function focuses on applying values robustly.)

    Parameters:
    - projection_df (pd.DataFrame): DataFrame containing one row per item, with Value_YEAR columns.
    - years (List[int]): List of target years (e.g., [2025, 2026, ..., 2035]).

    Returns:
    - pd.DataFrame: Updated projection DataFrame with adjusted Value_YEAR values (real terms).
    """

    if "growth_inputs" not in st.session_state:
        st.info("No growth or budget projections found.")
        return projection_df

    inflation_rates = st.session_state.get("inflation_rates", {})
    start_year = years[0]

    # --- Helper: cumulative inflation factor from (start_year+1) .. y ---
    def cumulative_inflation_factor(y: int) -> float:
        """
        Compute cumulative inflation factor from start_year (exclusive) up to y (inclusive).
        Example: if inflation is 2%, 1.5% -> factor = 1.02 * 1.015
        """
        factor = 1.0
        for yr in range(start_year + 1, y + 1):
            factor *= (1 + float(inflation_rates.get(yr, 0)) / 100.0)
        return factor

    for g in st.session_state.growth_inputs:
        selected_paths = set(g.get("categories", {}).get("checked", []))

        for idx, row in projection_df.iterrows():
            full_path = row["Full path"]
            if full_path not in selected_paths:
                continue

            base_val = row[f"Value_{start_year}"]

            # === Mode 1: Growth % ===
            if g["mode"] == "Growth %":
                nominal_growth = float(g["growth"])
                for y in years[1:]:
                    # Nominal compounding since start_year
                    nominal_factor = (1 + nominal_growth / 100.0) ** (y - start_year)
                    # Remove inflation to get real factor
                    real_factor = nominal_factor / cumulative_inflation_factor(y)
                    projection_df.at[idx, f"Value_{y}"] = base_val * real_factor

            # === Mode 2: Budget Projection (FIXED) ===
            elif g["mode"] == "Budget Projection":
                # 1) Normalize budget dict to {int_year: float_value}
                raw_budget = g.get("budget", {}) or {}
                budget = {}
                for k, v in raw_budget.items():
                    # keys can be '2025', 2025, '2025.0' -> normalize to int
                    try:
                        y_int = int(float(k))
                        budget[y_int] = float(v)
                    except (ValueError, TypeError):
                        continue  # ignore malformed entries

                if not budget:
                    # Nothing usable; skip safely
                    continue

                # 2) Validate presence and positivity of start-year budget
                if start_year not in budget:
                    st.warning(
                        f"Budget projection '{g.get('name','')}' has no value for start year {start_year}. "
                        f"Skipping application for selected rows."
                    )
                    continue

                start_budget = budget[start_year]
                if start_budget <= 0:
                    st.warning(
                        f"Budget projection '{g.get('name','')}' has start-year budget = 0 for {start_year}. "
                        f"Cannot scale values (division by zero). Skipping."
                    )
                    continue

                # 3) Prepare sorted known budget years
                known_years = sorted(y for y, val in budget.items() if val is not None)

                # 4) Helper to get a budget for ANY year y:
                #    - exact if provided
                #    - otherwise linear interpolation between nearest known years
                #    - if outside range: flat extrapolation using nearest known (conservative)
                def budget_for_year(y: int) -> float:
                    if y in budget:
                        return float(budget[y])

                    # Find nearest known years around y
                    before = [ky for ky in known_years if ky < y]
                    after = [ky for ky in known_years if ky > y]

                    if before and after:
                        y0 = max(before)
                        y1 = min(after)
                        v0 = float(budget[y0])
                        v1 = float(budget[y1])
                        # Linear interpolation
                        t = (y - y0) / (y1 - y0)
                        return v0 + t * (v1 - v0)

                    # Flat extrapolation if only one side exists
                    if before:
                        return float(budget[max(before)])
                    if after:
                        return float(budget[min(after)])

                    # Fallback (shouldn't happen if budget non-empty)
                    return float(start_budget)

                # 5) Apply real scaling to every target year
                for y in years:
                    # Compute real ratio = (Budget_y / Budget_start) / cumulative_inflation
                    B_y = budget_for_year(y)
                    real_ratio = (B_y / start_budget) / cumulative_inflation_factor(y)
                    projection_df.at[idx, f"Value_{y}"] = base_val * real_ratio

    return projection_df






def check_projection_coverage(projected_df):
    """
    Verify that each row in the projection table is covered by exactly one growth or budget projection.

    This function checks whether every item in the projected DataFrame is associated with
    a single projection based on the assigned category tree. It identifies two types of issues:
    - Rows without any assigned projection.
    - Rows matched by multiple projections.

    Parameters:
    - projected_df (pd.DataFrame): DataFrame containing the projection data, including a 'Full path' column.

    Effects:
    - Displays Streamlit warnings for rows with missing or overlapping projections.
    - Shows a maximum of 30 individual warnings for readability.
    - Displays a success message if all rows are correctly covered.
    """
    warnings = []

    if "growth_inputs" not in st.session_state:
        return

    for idx, row in projected_df.iterrows():
        full_path = row["Full path"]
        name = row["Name"]

        matching_growths = 0

        for g in st.session_state.growth_inputs:
            selected = set(g.get("categories", {}).get("checked", []))
            if full_path in selected:
                matching_growths += 1

        if matching_growths == 0:
            warnings.append(f"⚠️ No projection applied to: **{full_path} > {name}**")
        elif matching_growths > 1:
            warnings.append(f"⚠️ Multiple projections applied to: **{full_path} > {name}**")

    if warnings:
        st.warning("Some rows have missing or conflicting projections:")
        for w in warnings[:30]:  # Limit display
            st.markdown(f"- {w}")
        if len(warnings) > 30:
            st.markdown(f"...and {len(warnings) - 30} more.")
    else:
        st.success("✅ All rows are correctly covered by exactly one growth/budget projection.")
