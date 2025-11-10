import streamlit as st
import numpy as np
import pandas as pd
import graphviz
from copy import deepcopy
from math import inf

# -----------------------
# Helper data structures
# -----------------------
def _new_node_id(state):
    nid = state.get("tsp_next_id", 0)
    state["tsp_next_id"] = nid + 1
    return nid

# -----------------------
# Helper functions
# -----------------------
def reduce_matrix(matrix):
    """
    Reduce matrix: subtract row minima then column minima.
    Returns (reduced_matrix, reduction_cost, row_mins, col_mins).
    """
    m = matrix.astype(float).copy()
    n = m.shape[0]
    row_mins = np.zeros(n)
    col_mins = np.zeros(n)

    # Row reduction
    for i in range(n):
        valid = m[i][m[i] != np.inf]
        if valid.size > 0:
            rmin = valid.min()
            if rmin != np.inf and rmin > 0:
                row_mins[i] = rmin
                m[i] = np.where(m[i] != np.inf, m[i] - rmin, np.inf)

    # Column reduction
    for j in range(n):
        valid = m[:, j][m[:, j] != np.inf]
        if valid.size > 0:
            cmin = valid.min()
            if cmin != np.inf and cmin > 0:
                col_mins[j] = cmin
                m[:, j] = np.where(m[:, j] != np.inf, m[:, j] - cmin, np.inf)

    total_cost = row_mins.sum() + col_mins.sum()
    return m, total_cost, row_mins, col_mins

def branch_node(node, distance_matrix, state):
    """
    Expand `node` (dict) into child nodes using branch-and-bound rules.
    Each child is a dict with keys: id, parent, path, level, bound, matrix, status.
    """
    children = []
    n = distance_matrix.shape[0]
    current = node["path"][-1]
    for next_city in range(n):
        if next_city in node["path"]:
            continue
        # edge cost from original matrix
        edge_cost = distance_matrix[current, next_city]
        if edge_cost == np.inf:
            # unreachable
            continue
        # Build child matrix: copy node matrix, set row current and col next_city to inf
        temp = node["matrix"].copy()
        temp[current, :] = np.inf
        temp[:, next_city] = np.inf
        # forbid returning directly to start prematurely
        temp[next_city, node["path"][0]] = np.inf
        reduced, red_cost, row_mins, col_mins = reduce_matrix(temp)
        child_bound = node["bound"] + edge_cost + red_cost
        child = {
            "id": _new_node_id(state),
            "parent": node["id"],
            "path": node["path"] + [next_city],
            "level": node["level"] + 1,
            "bound": child_bound,
            "matrix": reduced,
            "orig_matrix": temp,  # prior to reduction (for display)
            "row_mins": row_mins,
            "col_mins": col_mins,
            "edge_cost": edge_cost,
            "status": "waiting"  # waiting/active/pruned/explored/solution
        }
        children.append(child)
    return children

def draw_tree(nodes):
    """
    Build a Graphviz dot object for the current node list.
    Color convention:
      - current (active) -> green
      - pruned -> red
      - waiting/frontier -> blue
      - explored -> gray
      - solution -> gold
    """
    dot = graphviz.Digraph(format="png")
    for node in nodes:
        nid = str(node["id"])
        label = f'{ "→".join(str(x) for x in node["path"]) }\\nLB={node["bound"]:.1f}'
        color = "lightblue"
        if node["status"] == "active":
            color = "green"
        elif node["status"] == "pruned":
            color = "red"
        elif node["status"] == "waiting":
            color = "deepskyblue"
        elif node["status"] == "explored":
            color = "gray"
        elif node["status"] == "solution":
            color = "gold"
        dot.node(nid, label=label, style="filled", fillcolor=color)
        if node["parent"] is not None:
            dot.edge(str(node["parent"]), nid)
    return dot

# -----------------------
# UI & algorithm control
# -----------------------
def _reset_state():
    st.session_state["tsp_initialized"] = False
    st.session_state["tsp_nodes"] = []
    st.session_state["tsp_frontier"] = []
    st.session_state["tsp_best_cost"] = np.inf
    st.session_state["tsp_best_path"] = None
    st.session_state["tsp_next_id"] = 0
    st.session_state["tsp_step"] = 0
    # last expansion info
    st.session_state.pop("tsp_last_children", None)
    st.session_state.pop("tsp_last_level_min", None)
    st.session_state.pop("tsp_last_expanded", None)

def show_tsp_bb():
    st.title("TSP - Branch and Bound (step-by-step)")

    # ----- Input Section -----
    st.header("A) Input")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.write("Provide a distance matrix (use ∞ for no edge or leave blank).")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded, header=None)
            distance_matrix = df.replace(["∞", "inf", "INF"], np.inf).astype(float).values
            st.session_state["tsp_distance_matrix"] = distance_matrix
            st.session_state["tsp_n"] = distance_matrix.shape[0]
        else:
            # editable matrix input
            n = st.number_input("Matrix size n (cities)", min_value=2, max_value=12, value=4, key="tsp_n_input")
            if "tsp_distance_matrix" not in st.session_state or st.session_state.get("tsp_n", None) != n:
                # initialize default matrix
                default = np.full((n, n), np.inf)
                np.fill_diagonal(default, 0.0)
                st.session_state["tsp_distance_matrix"] = default
                st.session_state["tsp_n"] = n
            # present data editor
            df_edit = pd.DataFrame(
                st.session_state["tsp_distance_matrix"]
            )
            edited = st.data_editor(df_edit, num_rows="fixed", use_container_width=True, key="tsp_data_editor")
            # convert blanks or non-numeric to inf
            def _to_float_or_inf(x):
                try:
                    if pd.isna(x): return np.inf
                    return float(x)
                except:
                    return np.inf
            distance_matrix = edited.applymap(_to_float_or_inf).values
            st.session_state["tsp_distance_matrix"] = distance_matrix
            st.session_state["tsp_n"] = distance_matrix.shape[0]

    with col_right:
        st.write("Options")
        directed = st.checkbox("Directed graph (asymmetric)", value=False, key="tsp_directed")
        prune_level_enabled = st.checkbox("Prune non-minimal children on each level", value=False, key="tsp_prune_level")
        st.write("Controls")
        if st.button("Initialize / Restart"):
            _reset_state()
            # prepare root
            distance_matrix = st.session_state["tsp_distance_matrix"].astype(float)
            n = distance_matrix.shape[0]
            # root reduction on original matrix
            root_reduced, red_cost, row_mins, col_mins = reduce_matrix(distance_matrix.copy())
            root = {
                "id": _new_node_id(st.session_state),
                "parent": None,
                "path": [0],
                "level": 0,
                "bound": red_cost,
                "matrix": root_reduced,
                "orig_matrix": distance_matrix.copy(),
                "row_mins": row_mins,
                "col_mins": col_mins,
                "status": "waiting"
            }
            st.session_state["tsp_nodes"] = [root]
            st.session_state["tsp_frontier"] = [root["id"]]
            st.session_state["tsp_distance_matrix"] = distance_matrix
            st.session_state["tsp_n"] = n
            st.session_state["tsp_initialized"] = True
            st.session_state["tsp_step"] = 0
            st.success("Initialized root node.")

    # show current matrix
    st.subheader("Distance Matrix")
    if "tsp_distance_matrix" in st.session_state:
        disp = pd.DataFrame(st.session_state["tsp_distance_matrix"])
        disp = disp.replace(np.inf, "∞")
        st.dataframe(disp)

    # ----- Controls for stepping -----
    st.header("Interactivity")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Next Step"):
            if not st.session_state.get("tsp_initialized", False):
                st.error("Initialize first.")
            else:
                _perform_next_step()
    with col2:
        if st.button("Finish (run to end)"):
            if not st.session_state.get("tsp_initialized", False):
                st.error("Initialize first.")
            else:
                # auto-run until done
                while _perform_next_step(auto=True):
                    pass

    with col3:
        if st.button("Clear / Reset"):
            _reset_state()
            st.experimental_rerun()

    # ----- Tree Visualization Section -----
    st.header("B) Tree Visualization")
    nodes = st.session_state.get("tsp_nodes", [])
    if nodes:
        dot = draw_tree(nodes)
        st.graphviz_chart(dot)

    # ----- Reduction Table Section -----
    st.header("C) Reduction Tables (current expansion)")
    cur_node = None
    # current node is the one with status 'active' or, if none, the next frontier minimal bound
    for n in st.session_state.get("tsp_nodes", []):
        if n["status"] == "active":
            cur_node = n
            break
    if cur_node is None:
        # pick minimal frontier for display only
        frontier_ids = st.session_state.get("tsp_frontier", [])
        frontier_nodes = [nd for nd in st.session_state.get("tsp_nodes", []) if nd["id"] in frontier_ids and nd["status"] != "pruned"]
        if frontier_nodes:
            cur_node = min(frontier_nodes, key=lambda x: x["bound"])
    if cur_node:
        st.write(f"Showing reductions for node id={cur_node['id']} path: {'→'.join(map(str, cur_node['path']))}")
        # original (with infinities applied for path)
        orig = cur_node.get("orig_matrix", cur_node["matrix"].copy())
        # show three columns: original(with infinities), after row reduction, after full reduction
        colA, colB, colC = st.columns(3)
        with colA:
            st.write("Original (masked) matrix")
            st.table(pd.DataFrame(orig).replace(np.inf, "∞"))
        # row reduced: apply row subtraction only
        temp = orig.copy()
        row_mins = np.zeros(temp.shape[0])
        for i in range(temp.shape[0]):
            valid = temp[i][temp[i] != np.inf]
            if valid.size > 0:
                rmin = valid.min()
                row_mins[i] = rmin
                if rmin != np.inf and rmin > 0:
                    temp[i] = np.where(temp[i] != np.inf, temp[i] - rmin, np.inf)
        with colB:
            st.write("After Row Reduction")
            st.table(pd.DataFrame(temp).replace(np.inf, "∞"))
            st.write("Row minima:", [float(x) for x in row_mins])
        # column reduction on top of row-reduced
        final = temp.copy()
        col_mins = np.zeros(final.shape[0])
        for j in range(final.shape[0]):
            valid = final[:, j][final[:, j] != np.inf]
            if valid.size > 0:
                cmin = valid.min()
                col_mins[j] = cmin
                if cmin != np.inf and cmin > 0:
                    final[:, j] = np.where(final[:, j] != np.inf, final[:, j] - cmin, np.inf)
        with colC:
            st.write("After Column Reduction")
            st.table(pd.DataFrame(final).replace(np.inf, "∞"))
            st.write("Column minima:", [float(x) for x in col_mins])
        lb = row_mins.sum() + col_mins.sum()
        st.write(f"Computed lower bound contribution from reductions: **{lb:.2f}**")

        # If we just expanded this node, show child bounds and level-min info
        last_expanded = st.session_state.get("tsp_last_expanded")
        if last_expanded == cur_node["id"]:
            last_children = st.session_state.get("tsp_last_children", [])
            if last_children:
                st.write("Children created in last expansion (path → bound → status):")
                for ch in last_children:
                    st.write(f"{'→'.join(map(str,ch['path']))}  —  LB={ch['bound']:.2f}  —  {ch['status']}")
                level_min = st.session_state.get("tsp_last_level_min", None)
                if level_min is not None:
                    st.info(f"Minimum LB at this level (rsl) = {level_min:.2f}")

    # ----- Final Output Section -----
    st.header("Final Output")
    if st.session_state.get("tsp_best_path") is not None:
        st.success("Optimal tour found!")
        st.metric("Optimal Cost", f"{st.session_state['tsp_best_cost']:.2f}")
        st.metric("Total Nodes Explored", len(st.session_state.get("tsp_nodes", [])))
        pruned = sum(1 for nd in st.session_state.get("tsp_nodes", []) if nd["status"] == "pruned")
        st.metric("Pruned Nodes", pruned)
        st.write("Optimal Path:", " → ".join(str(x) for x in st.session_state["tsp_best_path"]))

# -----------------------
# Core step function
# -----------------------
def _perform_next_step(auto=False):
    """
    Perform a single branch-and-bound expansion step.
    Returns True if there are more steps to process (used by Finish loop).
    """
    state = st.session_state
    nodes = state.get("tsp_nodes", [])
    if not nodes:
        return False

    # pick frontier nodes (waiting) with smallest bound
    frontier_ids = state.get("tsp_frontier", [])
    frontier_nodes = [nd for nd in nodes if nd["id"] in frontier_ids and nd["status"] != "pruned"]
    if not frontier_nodes:
        # no frontier: finish
        return False

    # choose node with minimum bound
    current = min(frontier_nodes, key=lambda x: x["bound"])
    # mark active
    for nd in nodes:
        if nd["status"] == "active":
            nd["status"] = "explored"
    current["status"] = "active"
    state["tsp_step"] = state.get("tsp_step", 0) + 1

    # If node level = n-1, it's a complete path (except return to start)
    n = state["tsp_n"]
    orig_matrix = state["tsp_distance_matrix"]
    if current["level"] == n - 1:
        # form full tour and compute cost using original matrix
        path = current["path"] + [current["path"][0]]
        total_cost = 0.0
        feasible = True
        for i in range(len(path) - 1):
            c = orig_matrix[path[i], path[i+1]]
            if c == np.inf:
                feasible = False
                break
            total_cost += c
        if feasible and total_cost < state.get("tsp_best_cost", np.inf):
            state["tsp_best_cost"] = total_cost
            state["tsp_best_path"] = path
            current["status"] = "solution"
        else:
            current["status"] = "explored"
        # remove from frontier
        state["tsp_frontier"] = [fid for fid in frontier_ids if fid != current["id"]]
        return len(state.get("tsp_frontier", [])) > 0

    # Otherwise expand current
    children = branch_node(current, orig_matrix, state)

    # compute level minimum (rsl) among children (bound includes edge + reductions)
    if children:
        level_min = min(ch["bound"] for ch in children)
    else:
        level_min = None

    # store last expansion info for UI display
    state["tsp_last_expanded"] = current["id"]
    state["tsp_last_level_min"] = level_min
    state["tsp_last_children"] = [{"path": ch["path"], "bound": ch["bound"], "status": ch["status"]} for ch in children]

    # add children: apply pruning rules
    prune_level_enabled = state.get("tsp_prune_level", False)
    for ch in children:
        # prune if worse than current best known solution
        if ch["bound"] >= state.get("tsp_best_cost", np.inf):
            ch["status"] = "pruned"
        else:
            # optional additional pruning: prune children on this level that are not minimal
            if prune_level_enabled and (level_min is not None) and (ch["bound"] > level_min):
                ch["status"] = "pruned"
            else:
                ch["status"] = "waiting"
                state.setdefault("tsp_frontier", []).append(ch["id"])
        nodes.append(ch)

    # mark current explored and remove from frontier
    current["status"] = "explored"
    state["tsp_frontier"] = [fid for fid in state.get("tsp_frontier", []) if fid != current["id"]]

    # update lists
    state["tsp_nodes"] = nodes

    # when auto is True, allow small progress without hanging UI
    if auto:
        return True
    return len(state.get("tsp_frontier", [])) > 0

