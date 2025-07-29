# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 07:34:46 2025

@author: quent
"""
import streamlit as st
import pandas as pd
from streamlit_tree_select import tree_select
#build_tree creates a tree with all the category and the subcategory to enable the user to select some of the categories for the growth, the structural effect or the solutions
#for now the function cant deal with None so we have to make sure than the category and sub 1 category is fully complete 

def build_tree(data):
    """
    Build a hierarchical tree from Category down to Localisation,
    dynamically including Sub-categories and skipping missing levels.
    The tree will have between 4 and 6 levels depending on the data.
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



# tree_growth_assignment enables the user to select the assignment thanks to the tree created with the previous function
# I dont use this function because i have included it in the next one but I keep it here for now

def tree_growth_assignment(data):
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
    Construct a full hierarchical label from a row, skipping empty or NaN fields.
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
    Initialize the projection base with columns Value_YEAR and EF_YEAR for each year in the range.

    Parameters:
    - data (DataFrame): The original dataset with columns including Category, Sub-category 1..3, Name, Location, Value, EF Value
    - years (List[int]): List of years for which to create projection columns

    Returns:
    - DataFrame: Prepared projection DataFrame with per-year value and EF columns, plus a 'Full path' column
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