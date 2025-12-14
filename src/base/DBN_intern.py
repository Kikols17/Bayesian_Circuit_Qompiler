from typing import List, Dict, Tuple, Any, Optional, Set
import itertools

from .DBN_Node_intern import DBN_Node_intern
from .CPT_intern import CPT_intern

class DBN_intern:
    """
    Internal representation of a Discrete Bayesian Network.
    """

    nodes: Dict[str, DBN_Node_intern] = {}          # 
    edges: Set[Tuple[str, str]] = set()             # 

    def __init__(self,
                 nodes: Optional[Dict[str, DBN_Node_intern]] = None,
                 edges: Optional[Set[Tuple[str, str]]] = None
                ) -> None:
        # Use provided containers or default to empty ones (avoid shared mutable defaults)
        self.nodes = nodes if nodes is not None else {}
        self.edges = edges if edges is not None else set()

    def add_node(self,
                 name: str,
                 node: DBN_Node_intern
                ) -> None:
        """
        Adds a node to the network.
        :param name: Name of the node (string)
        :param node: Node object (DBN_Node_intern)
        """
        if name in self.nodes:
            raise ValueError(f"Node name \"{name}\" already exists.")
        if node in self.nodes.values():
            raise ValueError(f"Node object \"{node}\" already exists.")
        # Initialize with empty evidences
        self.nodes[name] = node

    def add_edge(self,
                 parent: str,
                 child: str,
                 new_cptentries: Dict[Tuple[str|bool, ...], float]
                ) -> None:
        """
        Adds a directed edge from parent to child and integrates the provided CPT entries.
        :param parent: Name of the parent node
        :param child: Name of the child node
        :param new_cptentries: Full CPT for the child after adding this parent. Keys must be
                               tuples of the form (child_state, *evidence_states) where the
                               evidence_states follow the order in child_node.evidences
        """
        # verify node's validity
        if parent not in self.nodes:
            raise ValueError(f"Parent node {parent} does not exist.")
        if child not in self.nodes:
            raise ValueError(f"Child node {child} does not exist.")

        # verify edge validity (maintain acyclic)
        if (parent, child) in (self.edges or set()):
            raise ValueError(f"Edge \"{parent}\"->\"{child}\" already exists.")
        # If there's already a path from "child" back to "parent", adding parent->child creates a cycle.
        def _path_exists(start: str, goal: str) -> bool:
            stack = [start]
            visited = set()
            while stack:
                node = stack.pop()
                if node == goal:
                    return True
                if node in visited:
                    continue
                visited.add(node)
                for (u, v) in self.edges or set():
                    if u == node and v not in visited:
                        stack.append(v)
            return False

        if _path_exists(child, parent):
            raise ValueError(f"Adding edge \"{parent}\"->\"{child}\" would create a cycle.")

        # add the edge
        if self.edges is None:
            self.edges = set()
        self.edges.add((parent, child))

        # Update child's evidence info
        child_node = self.nodes[child]
        parent_node = self.nodes[parent]

        if child_node.evidences is None:
            child_node.evidences = []
        if child_node.evidences_states is None:
            child_node.evidences_states = {}

        if parent not in child_node.evidences:
            child_node.evidences.append(parent)
            child_node.evidences_states[parent] = parent_node.states

        # Validate and integrate new_cptentries
        # Prepare expected evidence ordering and state lists
        evidences = child_node.evidences or []
        evidences_state_lists = [child_node.evidences_states[e] for e in evidences]

        # Basic expectations
        child_states = child_node.states
        if child_states is None:
            raise ValueError(f"Child node {child} has no defined states.")

        expected_num = len(child_states)
        for lst in evidences_state_lists:
            expected_num *= len(lst)

        if len(new_cptentries) != expected_num:
            raise ValueError(f"Expected {expected_num} CPT entries for node '{child}' after adding parent "
                             f"'{parent}', but got {len(new_cptentries)}.")

        # Validate keys and values, and ensure coverage of all evidence combinations
        # Expected keys are tuples of length 1 + len(evidences): (child_state, *evidence_values)
        expected_ev_combinations = set(itertools.product(*evidences_state_lists)) if evidences_state_lists else {()}
        seen_ev_combinations = set()

        # Validate each entry
        for k, prob in new_cptentries.items():
            if not isinstance(k, tuple):
                raise ValueError("CPT key must be a tuple: (child_state, *evidence_states).")
            if len(k) != 1 + len(evidences):
                raise ValueError(f"CPT key {k} has wrong length; expected {1 + len(evidences)} elements.")
            var_state = k[0]
            if var_state not in child_states:
                raise ValueError(f"CPT key {k}: unknown child state '{var_state}' (expected one of {child_states}).")
            ev_vals = k[1:]
            # validate each evidence value
            for ev_name, ev_val in zip(evidences, ev_vals):
                if ev_val not in child_node.evidences_states[ev_name]:
                    raise ValueError(f"CPT key {k}: value '{ev_val}' is not a valid state for evidence '{ev_name}'.")
            seen_ev_combinations.add(ev_vals)
            # validate probability value type/range
            if not isinstance(prob, (int, float)):
                raise ValueError(f"CPT value for key {k} must be numeric.")
            if prob < 0 or prob > 1:
                raise ValueError(f"CPT probability for key {k} must be in [0,1], got {prob}.")

        if seen_ev_combinations != expected_ev_combinations:
            missing = expected_ev_combinations - seen_ev_combinations
            extra = seen_ev_combinations - expected_ev_combinations
            raise ValueError(f"CPT evidence combinations do not match expected set. "
                             f"Missing combinations: {missing}. Extra combinations: {extra}.")

        # Check normalization: for each evidence combination, probabilities over child states should sum to 1
        tol = 1e-8
        for ev_comb in expected_ev_combinations:
            s = 0.0
            for st in child_states:
                key = (st,) + ev_comb
                s += float(new_cptentries[key])
            if abs(s - 1.0) > tol:
                raise ValueError(f"Probabilities for evidence combination {ev_comb} do not sum to 1 (sum={s}).")

        # Create and assign new CPT object for the child
        cpt = CPT_intern(
            variable=child,
            variable_states=child_node.states,
            evidences=evidences,
            evidences_states=child_node.evidences_states,
            table=new_cptentries
        )
        child_node.cpt = cpt

    def set_cpt(self, node_name: str, table: Dict[Tuple[Any, ...], float]):
        """
        Sets the Conditional Probability Table for a node.
        :param node_name: Name of the node
        :param table: The CPT data structure as a dictionary {(val, p1_val, ...): prob}
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist.")

        node = self.nodes[node_name]

        # Create CPT
        cpt = CPT_intern(
            variable=node_name,
            variable_states=node.states,
            evidences=node.evidences,
            evidences_states=node.evidences_states,
            table=table
        )
        node.cpt = cpt

    def get_node(self, node_name: str) -> Optional[DBN_Node_intern]:
        """Returns the DBN_Node object."""
        return self.nodes.get(node_name)

    def get_parents(self, node_name: str) -> List[str]:
        """Returns the list of parents for a node (names)."""
        return [p for p, c in self.edges if c == node_name]

    def get_children(self, node_name: str) -> List[str]:
        """Returns the list of children for a node (names)."""
        return [c for p, c in self.edges if p == node_name]

    def get_states(self, node_name: str) -> Optional[List[str | bool]]:
        """Returns the states of a node."""
        node = self.nodes.get(node_name)
        if node:
            return node.states
        return None

    def get_cpt(self, node_name: str) -> Optional[CPT_intern]:
        """Returns the CPT object of a node."""
        node = self.nodes.get(node_name)
        if node:
            return node.cpt
        return None

    def __repr__(self):
        return f"DBN(nodes={list(self.nodes.keys())}, edges={len(self.edges)})"

    def to_pgmpy(self):
        """
        Converts the DBN_intern model to a pgmpy BayesianNetwork.
        Requires pgmpy to be installed.
        """
        try:
            from pgmpy.models import BayesianNetwork
            from pgmpy.factors.discrete import TabularCPD
        except ImportError:
            raise ImportError("pgmpy is not installed. Please install it to use this feature.")

        model = BayesianNetwork(self.edges)
        model.add_nodes_from(self.nodes.keys())

        for name, node in self.nodes.items():
            if node.cpt is None:
                continue

            variable_card = len(node.states)
            evidence = node.evidences
            evidence_card = [len(node.evidences_states[e]) for e in evidence]

            # Generate all combinations of evidence states
            # pgmpy expects the last evidence in the list to vary fastest
            evidence_state_lists = [node.evidences_states[e] for e in evidence]
            evidence_combinations = list(itertools.product(*evidence_state_lists))

            values = []
            for var_state in node.states:
                # For each state of the variable, we append a row of probabilities
                # corresponding to all evidence combinations
                for ev_comb in evidence_combinations:
                    # Key for CPT_intern is (var_state, *ev_comb)
                    key = (var_state,) + ev_comb
                    # We use get_probability to handle potential missing keys if CPT is sparse (though pgmpy needs full)
                    try:
                        prob = node.cpt.get_probability(var_state, ev_comb)
                    except KeyError:
                        prob = 0.0 # Or raise error?
                    values.append(prob)

            cpd = TabularCPD(
                variable=name,
                variable_card=variable_card,
                values=[values], # pgmpy accepts flat list
                evidence=evidence,
                evidence_card=evidence_card,
                state_names={
                    name: node.states,
                    **{e: node.evidences_states[e] for e in evidence}
                }
            )
            model.add_cpds(cpd)

        return model

    @classmethod
    def from_pgmpy(cls, model):
        """
        Creates a DBN_intern model from a pgmpy BayesianNetwork.
        """
        dbn = cls()

        # Add nodes
        for node_name in model.nodes():
            try:
                cpd = model.get_cpds(node_name)
                states = cpd.state_names[node_name]
            except:
                # Fallback if no CPD is defined, try to infer or raise error
                # For now, we require CPDs to define states
                raise ValueError(f"Node {node_name} has no CPD or states defined in pgmpy model.")

            dbn.add_node(node_name, states)

        # Add edges
        for u, v in model.edges():
            dbn.add_edge(u, v)

        # Set CPTs
        for node_name in model.nodes():
            cpd = model.get_cpds(node_name)
            if cpd:
                var_states = cpd.state_names[node_name]
                # cpd.variables[0] is the variable itself
                # cpd.variables[1:] are the evidences
                evidence_vars = cpd.variables[1:]
                evidence_state_lists = [cpd.state_names[e] for e in evidence_vars]

                # Flattened values iteration
                # pgmpy values are stored such that last evidence varies fastest
                evidence_combinations = list(itertools.product(*evidence_state_lists))

                table = {}
                # cpd.values is usually (var_card, num_columns)
                # We can iterate rows (var states) and cols (evidence combs)

                # Ensure values are in 2D shape (var_card, product_of_evidence_cards)
                values_2d = cpd.values.reshape(len(var_states), -1)

                for i, var_state in enumerate(var_states):
                    for j, ev_comb in enumerate(evidence_combinations):
                        prob = float(values_2d[i, j])
                        key = (var_state,) + ev_comb
                        table[key] = prob

                dbn.set_cpt(node_name, table)

        return dbn
