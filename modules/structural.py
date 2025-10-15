# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:39:58 2025

@author: quent
"""
import streamlit as st
from streamlit_tree_select import tree_select
from modules.tree import build_tree

def init_structural_effects():
    """
    Initialise the default list of structural effects in session state if not already defined.
    
    Structural effects are predefined emission multipliers applied to specific categories 
    (e.g. electricity, aviation). Each effect includes a name, a multiplier value, and a list
    of associated categories (initially empty).
    
    Effects:
    - Adds a 'structural_effects' key to st.session_state if it does not exist.
    - The default list includes predefined effects with typical emission multipliers.
    """
    
    if "structural_effects" not in st.session_state:
        st.session_state.structural_effects = [
            {"name": "Electricity from the grid", "value": -1.2, "categories": []},
            {"name": "Aviation", "value": -2.0, "categories": []},
            {"name": "International maritime transport", "value": -1.0, "categories": []},
            {"name": "Procurement of goods", "value": -3.4, "categories": []},
            {"name": "Procurement of services", "value": -2.3, "categories": []}
        ]


def create_structural_effect():
    """
    Display a form to create a new structural effect and store it in session state.

    A structural effect represents an annual percentage change in emission factors 
    (e.g. -2 for -2%/year). Positive values indicate increases, negative values indicate reductions.

    Effects:
    - Displays a form to enter the effect name and annual percentage.
    - Appends the new effect to st.session_state["structural_effects"] with an empty category selection.
    - Shows a confirmation message upon successful creation.
    """

    st.subheader("Create a new structural effect")

    # 🛈 Helper note
    st.markdown(
        """
        <div style='color: grey; font-size: 0.9em;'>
        💡 Enter an <strong>annual change percentage</strong> to represent how emission factors evolve over time.<br>
        For example, <strong>-2</strong> means a <strong>2% annual reduction</strong>, while <strong>+3</strong> means a <strong>3% annual increase</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("form_create_structural_effect"):
        name = st.text_input("Name of the structural effect")
        percentage = st.number_input(
            "Annual change (%) — e.g. -2 for -2%/year, +3 for +3%/year",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            format="%.2f"
        )
        submitted = st.form_submit_button("Add effect")

        if submitted and name:
            new_effect = {
                "name": name,
                "value": percentage,  # stored as % (not multiplier)
                "categories": {"checked": [], "expanded": []}
            }

            if "structural_effects" not in st.session_state:
                st.session_state.structural_effects = []

            st.session_state.structural_effects.append(new_effect)
            st.success(f"✅ Structural effect '{name}' added ({percentage:+.2f}%/year).")




def assign_structural_effects(data):
    """
    Display and manage structural effects, allowing category assignment, edition, and deletion.

    Parameters:
    - data (pd.DataFrame): DataFrame used to construct the category hierarchy.

    Effects:
    - Displays all existing structural effects.
    - Allows users to adjust annual % changes, assign categories, and delete effects.
    - Updates st.session_state["structural_effects"] accordingly.
    """
    st.subheader("Assign structural effects to categories")

    # 🛈 Helper paragraph for guidance
    st.markdown(
        """
        <div style='color: grey; font-size: 0.9em;'>
        💡 You can <strong>delete a structural effect</strong> if it's no longer relevant.<br>
        To do so, first click <strong>🗑️ Delete</strong> — this will remove its assigned categories.<br>
        Then click <strong>Save configuration</strong> to confirm and permanently remove it from the list.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "structural_effects" not in st.session_state or not st.session_state.structural_effects:
        st.info("No structural effects defined yet.")
        return

    tree = build_tree(data)
    cols = st.columns(3)
    to_delete = []  # Track effects to delete after rendering

    import re

    for i, effect in enumerate(st.session_state.structural_effects):
        form_id = re.sub(r"\W+", "_", effect["name"])
        col = cols[i % 3]

        with col.form(f"form_edit_structural_{form_id}"):
            st.markdown(f"### ⚙️ `{effect['name']}`")

            new_value = st.number_input(
                "Annual change (%) — e.g. -2 for -2%/year, +3 for +3%/year",
                min_value=-100.0,
                max_value=100.0,
                value=effect.get("value", 0.0),
                format="%.2f",
                key=f"percent_{effect['name']}"
            )

            selection = tree_select(
                tree,
                checked=effect["categories"]["checked"] if isinstance(effect.get("categories"), dict) else effect.get("categories", []),
                expanded=effect["categories"]["expanded"] if isinstance(effect.get("categories"), dict) else [],
                key=f"tree_struct_{effect['name']}"
            )

            # Update state before save
            st.session_state.structural_effects[i]["value"] = new_value
            st.session_state.structural_effects[i]["categories"] = selection

            # Two-column layout for buttons
            col_save, col_del = st.columns([3, 1])
            with col_save:
                submitted = st.form_submit_button("Save configuration")
            with col_del:
                delete_clicked = st.form_submit_button("🗑️ Delete", type="secondary")

            if submitted:
                st.success(f"✅ Configuration for '{effect['name']}' saved.")
            if delete_clicked:
                to_delete.append(i)

    # Delete after loop (avoid concurrent modification)
    if to_delete:
        for idx in sorted(to_delete, reverse=True):
            deleted_name = st.session_state.structural_effects[idx]["name"]
            del st.session_state.structural_effects[idx]
            st.warning(f"🗑️ Structural effect '{deleted_name}' deleted.")






def apply_structural_effects(data):
    """
    Apply structural effects to Emission Factor (EF) columns in the projection DataFrame.

    Each effect now represents an **annual percentage change** rather than a multiplier.
    For example, -2 means a 2% annual decrease in emission factors.

    For each row, the function identifies applicable structural effects based on category
    assignment. The effects are applied cumulatively to EF columns year by year 
    (starting from the second year), using the previous year's value multiplied by 
    (1 + percentage/100) for each applicable effect.
    
    Parameters:
    - data (pd.DataFrame): Projection DataFrame containing EF_YEAR columns and a 'Full path' column.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with structural effects applied to EF_YEAR columns.
    """

    if "structural_effects" not in st.session_state:
        return data

    df = data.copy()
    ef_cols = sorted([col for col in df.columns if col.startswith("EF_")])

    for idx, row in df.iterrows():
        full_path = row["Full path"]

        # Identify all effects applicable to this row
        applicable_effects = []
        for effect in st.session_state.structural_effects:
            categories = effect.get("categories", {})

            # Extract checked categories
            if isinstance(categories, dict):
                checked = categories.get("checked", [])
            elif isinstance(categories, list):
                checked = categories
            else:
                checked = []

            if full_path in set(checked):
                applicable_effects.append(effect)

        if not applicable_effects:
            continue

        # Apply cumulative effects year by year
        for i, col in enumerate(ef_cols):
            if i == 0:
                continue  # EF_2025 remains unchanged (baseline year)

            prev_col = ef_cols[i - 1]
            new_val = df.at[idx, prev_col]

            for effect in applicable_effects:
                # Convert stored % to multiplier
                percent_change = effect.get("value", 0.0)
                multiplier = 1 + (percent_change / 100)
                new_val *= multiplier

            df.at[idx, col] = new_val

    return df



def check_structural_coverage(data):
    """
    Check that each Full path is affected by at most one structural effect.

    This function verifies that no row in the emissions dataset is assigned to
    multiple structural effects. If overlaps are found (i.e. a Full path appears
    in more than one effect), a warning is displayed listing all conflicts.

    Parameters:
    - data (pd.DataFrame): Dataset containing a 'Full path' column for each emission source.

    Effects:
    - Displays warnings in Streamlit for all rows affected by more than one structural effect.
    - Displays a success message if no overlaps are found.
    """
    
    if "structural_effects" not in st.session_state:
        return

    # Dictionnaire : clé = Full path, valeur = liste des effets qui s’y appliquent
    coverage = {}

    for effect in st.session_state.structural_effects:
        categories = effect.get("categories", {})

        # Gestion sécurisée du format
        if isinstance(categories, dict):
            checked = categories.get("checked", [])
        elif isinstance(categories, list):
            checked = categories
        else:
            checked = []

        for path in checked:
            if path not in coverage:
                coverage[path] = []
            coverage[path].append(effect.get("name", "Unnamed effect"))

    # Vérifie les doublons
    overlapping = {path: names for path, names in coverage.items() if len(names) > 1}

    if overlapping:
        st.warning("⚠️ Some rows are affected by more than one structural effect:")
        for path, effects in overlapping.items():
            st.markdown(f"- `{path}` ➤ {', '.join(effects)}")
    else:
        st.success("✅ Each Full path is covered by at most one structural effect.")


