# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:18:52 2025

@author: quent
"""

import streamlit as st

def tree_select(label, nodes, checked=[]):
    """
    Simple tree view with checkboxes.

    Args:
        label (str): The label to display.
        nodes (list): List of dicts representing the tree structure.
        checked (list): List of keys that are checked by default.

    Returns:
        List of checked keys.
    """
    st.write(label)

    def render_node(node):
        key = node.get("key")
        label = node.get("label")
        children = node.get("children", [])
        is_checked = key in checked

        if children:
            with st.expander(label, expanded=False):
                for child in children:
                    yield from render_node(child)
        else:
            checked_state = st.checkbox(label, value=is_checked, key=key)
            if checked_state:
                yield key

    checked_nodes = list(render_node({"children": nodes}))
    return checked_nodes
