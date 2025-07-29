# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:34:46 2025

@author: quent
"""
import streamlit as st
import pandas as pd
from streamlit_tree_select import tree_select

def build_tree(data):
    """
    Build a hierarchical tree from emission data, going from Category to Localisation.

    This function constructs a nested dictionary tree using the columns:
    Category, Sub-category 1, Sub-category 2, Sub-category 3, Name, and Localisation.
    Missing levels are automatically skipped, allowing the tree to adapt dynamically 
    between 4 and 6 levels depending on the data.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the hierarchical structure of emissions.

    Returns:
    - List[Dict]: A tree structure formatted as a list of nested dictionaries with 'label',
      'value', and optional 'children' keys, ready to be used in tree-based selection widgets.
    """
    
    hierarchy_cols = [
        "Category",
        "Sub-category 1",
        "Sub-category 2",
        "Sub-category 3",
        "Name",
        "Localisation"
    ]

    # Replace NaN with None to make the check easier
    data_clean = data[hierarchy_cols].fillna(value=pd.NA)

    tree = {}

    for _, row in data_clean.iterrows():
        current_level = tree
        path = []

        # Build the tree level by level
        for col in hierarchy_cols:
            value = row[col]
            if pd.isna(value):
                continue  # Skip missing levels

            value = str(value).strip()
            path.append(value)

            # Build a unique path for the current node
            label = value
            full_value = " > ".join(path)

            # If this level does not exist yet, create it
            if "_children" not in current_level:
                current_level["_children"] = {}

            if value not in current_level["_children"]:
                current_level["_children"][value] = {
                    "label": label,
                    "value": full_value
                }

            # Move to the next level
            current_level = current_level["_children"][value]

    # Convert the nested dictionary into a list-based tree
    def dict_to_tree(node):
        result = {
            "label": node["label"],
            "value": node["value"]
        }
        if "_children" in node:
            result["children"] = [
                dict_to_tree(child) for child in node["_children"].values()
            ]
        return result

    final_tree = [dict_to_tree(child) for child in tree.get("_children", {}).values()]

    # Optional: Preview JSON
    #st.subheader("🧪 Tree structure preview")
    #st.json(final_tree)

    return final_tree

def tree_growth_assignment(data):
    """
    Display a tree structure to assign growth projections to selected categories.
    
    For each growth input defined in session state, this function shows an expandable
    tree selector (based on the input data) allowing the user to assign the growth
    to specific categories.
    
    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the full emissions data used to build the tree.
    
    Effects:
    - Updates each item in st.session_state["growth_inputs"] by storing the selected tree path(s)
      under the key 'selection_tree'.
    - Displays one expandable section per growth entry with an interactive tree selector.
    """

    tree_data = build_tree(data)

    st.subheader("Assign growth to categories")
    
    for i, g in enumerate(st.session_state.growth_inputs):
        with st.expander(f"Assign to: {g['name']} ({g['growth']}%)"):
            # Explicit unique key
            selection = tree_select(
                tree_data,
                key=f"tree_selector_{i}"
            )
            
            # Save the result
            st.session_state.growth_inputs[i]["selection_tree"] = selection

            # Debug output (optional)
            # st.json(selection)

def get_label_path(row):
    """
    Build a full hierarchical label from a DataFrame row, skipping empty or NaN fields.
    
    This function concatenates all relevant category levels from a single row 
    into a single string separated by ' > ', omitting any missing values.
    
    Parameters:
    - row (pd.Series): A row from the emissions DataFrame.
    
    Returns:
    - str: A hierarchical label string representing the full path of the item.
    """
    parts = [
        row.get("Category"),
        row.get("Sub-category 1"),
        row.get("Sub-category 2"),
        row.get("Sub-category 3"),
        row.get("Name"),
        row.get("Localisation")
    ]
    return " > ".join(
        str(p).strip()
        for p in parts
        if pd.notna(p) and str(p).strip() != ""
    )


def create_projection_base(data, years):
    """
    Create a projection-ready DataFrame with yearly Value and Emission Factor columns.

    This function initializes a new DataFrame where each row is duplicated for all 
    target years, with columns 'Value_YEAR' and 'EF_YEAR' filled from the original data.
    A 'Full path' column is also generated for tree-based selection and traceability.

    Parameters:
    - data (pd.DataFrame): Original dataset containing Category, Sub-category 1 to 3,
      Name, Localisation, Value, and EF Value columns.
    - years (List[int]): List of target years (e.g., [2025, 2026, ..., 2035]).

    Returns:
    - pd.DataFrame: Projection-ready DataFrame with one row per item and year,
      containing 'Value_YEAR', 'EF_YEAR', and a 'Full path' column.
    """
    base = []

    for _, row in data.iterrows():
        entry = {
            "Category": row.get("Category", ""),
            "Sub-category 1": row.get("Sub-category 1", ""),
            "Sub-category 2": row.get("Sub-category 2", ""),
            "Sub-category 3": row.get("Sub-category 3", ""),
            "Name": row.get("Name", ""),
            "Localisation": row.get("Localisation", ""),
        }

        for year in years:
            entry[f"Value_{year}"] = row["Value"]
            entry[f"EF_{year}"] = row["EF Value"]

        base.append(entry)

    df = pd.DataFrame(base)

    # Create full path column (skip empty levels)
    df["Full path"] = df.apply(get_label_path, axis=1)

    return df