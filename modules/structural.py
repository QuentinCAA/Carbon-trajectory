# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:39:58 2025

@author: quent
"""
import streamlit as st
from streamlit_tree_select import tree_select
from modules.tree import build_tree

#add_structural_effect: to give the possibilty to the user to add effects if he needs to
#Choice made : work with % and only one % for each structural effect (no way yet to create a more complex effect)

#init_structural_effects : i have put the structural effects i remember but need a double check

def init_structural_effects():
    if "structural_effects" not in st.session_state:
        st.session_state.structural_effects = [
            {"name": "Electricity from the grid", "value": 1.13, "categories": []},
            {"name": "Aviation", "value": 2.0, "categories": []},
            {"name": "International maritime transport", "value": 2.0, "categories": []},
            {"name": "Procurement of goods", "value": 3.43, "categories": []},
            {"name": "Procurement of services", "value": 2.32, "categories": []}]
        


def create_structural_effect():
    """
    UI for creating a new structural effect with a name and multiplier.
    """
    st.subheader("Create a new structural effect")

    with st.form("form_create_structural_effect"):
        name = st.text_input("Name of the structural effect")
        value = st.number_input("Multiplier (e.g. 0.8 = 20% reduction)", min_value=0.0, value=1.0, format="%.2f")
        submitted = st.form_submit_button("Add effect")

        if submitted and name:
            new_effect = {
                "name": name,
                "value": value,
                "categories": {"checked": [], "expanded": []}
            }

            if "structural_effects" not in st.session_state:
                st.session_state.structural_effects = []

            st.session_state.structural_effects.append(new_effect)
            st.success(f"✅ Structural effect '{name}' added.")


def assign_structural_effects(data):
    """
    Display and configure existing structural effects,
    allowing category assignment via tree and editing the multiplier.
    """
    st.subheader("Assign structural effects to categories")

    if "structural_effects" not in st.session_state or not st.session_state.structural_effects:
        st.info("No structural effects defined yet.")
        return

    tree = build_tree(data)

    for i, effect in enumerate(st.session_state.structural_effects):
        # Clean up effect name for form ID
        import re
        form_id = re.sub(r"\W+", "_", effect["name"])

        with st.form(f"form_edit_structural_{form_id}"):
            st.markdown(f"### ⚙️ `{effect['name']}`")

            new_value = st.number_input(
                "Multiplier (e.g. 0.8 = 20% reduction)",
                min_value=0.0,
                value=effect.get("value", 1.0),
                format="%.2f",
                key=f"multiplier_{effect['name']}"
            )

            selection = tree_select(
                tree,
                checked=effect["categories"]["checked"] if isinstance(effect.get("categories"), dict) else effect.get("categories", []),
                expanded=effect["categories"]["expanded"] if isinstance(effect.get("categories"), dict) else [],
                key=f"tree_struct_{effect['name']}"
            )

            submitted = st.form_submit_button("Save configuration")

            if submitted:
                st.session_state.structural_effects[i]["value"] = new_value
                st.session_state.structural_effects[i]["categories"] = selection
                st.success(f"✅ Configuration for '{effect['name']}' saved.")




def apply_structural_effects(data):
    """
    Apply structural effects (percentage evolution of EF) to the projection DataFrame.
    Effects are applied cumulatively year after year to EF columns.
    """
    if "structural_effects" not in st.session_state:
        return data

    df = data.copy()
    ef_cols = sorted([col for col in df.columns if col.startswith("EF_")])

    for idx, row in df.iterrows():
        full_path = row["Full path"]

        applicable_effects = []
        for effect in st.session_state.structural_effects:
            categories = effect.get("categories", {})

            # If it's a dict: extract from "checked"
            if isinstance(categories, dict):
                checked = categories.get("checked", [])
            # If it's already a list: keep it as is
            elif isinstance(categories, list):
                checked = categories
            else:
                checked = []

            if full_path in set(checked):
                applicable_effects.append(effect)

        if not applicable_effects:
            continue

        for i, col in enumerate(ef_cols):
            if i == 0:
                continue  # EF_2025 remains unchanged

            prev_col = ef_cols[i - 1]
            new_val = df.at[idx, prev_col]

            for effect in applicable_effects:
                multiplier = effect.get("value", 1.0)
                new_val *= multiplier

            df.at[idx, col] = new_val

    return df


def check_structural_coverage(data):
    """
    Check that each Full path is affected by at most one structural effect.
    Warn if any row is affected by more than one.
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


