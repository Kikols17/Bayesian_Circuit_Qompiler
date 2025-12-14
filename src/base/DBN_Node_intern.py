from .CPT_intern import CPT_intern

from typing import (
    List,
    Dict,
    Optional
)


class DBN_Node_intern:
    """Internal representation of a Discrete Bayesian Network's Node"""
    name: str                                   # Node name
    states: List[str|bool]                      # Ordered list of states' names (bool for binary states, string otherwise)
    evidences: List[str]                        # Evidence variables' names
    evidences_states: Dict[str, List[str|bool]] # Dictionary that associates evidences' variable names to ordered lists of state names
    cpt = CPT_intern                            # CPT object

    def __init__(self,
                 name: Optional[str] = None,
                 states: Optional[List[str|bool]] = None,
                 evidences: Optional[List[str]] = None,
                 evidences_states: Optional[Dict[str, List[str|bool]]] = None,
                 cpt: Optional[CPT_intern] = None
                ) -> None:
        self.name = name
        self.states = states
        self.evidences = evidences
        self.evidences_states = evidences_states
        self.cpt = cpt

    def validate(self,
             tol: float = 1e-8
            ):
        # Ensure states, evidences, evidences_states and CPT are set before validation
        if self.states is None:
            raise ValueError("States are not set for this node")
        if self.cpt is None:
            raise ValueError("CPT is not set for this node")
        if self.evidences is None:
            raise ValueError("evidences is not set for this node")
        if self.evidences_states is None:
            raise ValueError("evidences_states is not set for this node")

        # Raise the exception upstream
        try:
            self.cpt.validate(tol=tol)
        except Exception as e:
            raise ValueError(f"CPT validation failed: {e}") from e

        # Checks if self.states == self.cpt.variable_states
        if self.cpt.variable_states != self.states:
            raise ValueError(
                f"Node states \"{self.states}\" do not match CPT.variable_states \"{self.cpt.variable_states}\#"
            )
        # Checks if self.evidences == self.cpt.evidences
        # and if self.evidences_states == self.cpt.evidences_states
        if len(self.evidences) != len(self.cpt.evidences):
            raise ValueError(f"Node evidences \"{self.evidences}\" size does not match CPT.evidences \"{self.cpt.evidences}\"")
        for evidence in self.evidences:
            if evidence not in self.cpt.evidences:
                raise ValueError(f"Node evidence \"{evidence}\" not in CPT.evidences \"{self.cpt.evidences}\"")
            for evidence_state in self.evidences_states[evidence]:
                if evidence_state not in self.cpt.evidences_states[evidence]:
                    raise ValueError(f"Node evidences's state \"{evidence_state}\" not in CPT.evidences \"{self.cpt.evidences_states[evidence]}\"")

    def __repr__(self) -> str:
        if self.cpt is not None:
            return f"DBN_Node_intern(name={self.name}, states={self.states}, cpt_size{self.cpt.get_tablesize()})"
        return f"DBN_Node_intern(name={self.name}, states={self.states})"


if __name__ == "__main__":
    # To run this example: python -m src.base.DBN_Node_intern
    print("--- Example 1: Simple Node Creation ---")
    node_a = DBN_Node_intern(name="A", states=["on", "off"])
    print(f"Created node: {node_a}")


    print("\n--- Example 2: Assign CPT to Node (no evidences) ---")
    # Create a CPT for A with no evidences
    cpt_a = CPT_intern(
        variable="A",
        variable_states=["on", "off"],
        evidences=[],
        evidences_states={},
        table={("on",): 0.7, ("off",): 0.3}
    )
    node_a.cpt = cpt_a
    # set evidences and evidences_states for node (empty)
    node_a.evidences = []
    node_a.evidences_states = {}
    print(f"Assigned CPT to node {node_a.name}")
    try:
        node_a.validate()
        print("Validation passed.")
    except ValueError as e:
        print(f"Validation failed: {e}")


    print("\n--- Example 3: Full Node Creation (with CPT, no evidences) ---")
    node_b = DBN_Node_intern(name="A",
                             states=["on", "off"],
                             evidences=[],
                             evidences_states={},
                             cpt=CPT_intern(
                                variable="A",
                                variable_states=["on", "off"],
                                evidences=[],
                                evidences_states={},
                                table={("on",): 0.7, ("off",): 0.3}
                             )
                            )
    print(f"Created node: {node_b} with CPT")
    try:
        print(f"{node_b.cpt.table_string()}")
    except Exception:
        pass
    try:
        node_b.validate()
        print("Validation passed.")
    except ValueError as e:
        print(f"Validation failed: {e}")



    print("\n--- Example 4: Node with one evidence (valid evidences_states) ---")
    # Node A depends on evidence B which is binary: "true"/"false"
    cpt_with_evidence = CPT_intern(
        variable="A",
        variable_states=["on", "off"],
        evidences=["B"],
        evidences_states={"B": ["true", "false"]},
        table={
            ("on", "true"): 0.8, ("off", "true"): 0.2,
            ("on", "false"): 0.6, ("off", "false"): 0.4
        }
    )
    node_d = DBN_Node_intern(
        name="A",
        states=["on", "off"],
        evidences=["B"],
        evidences_states={"B": ["true", "false"]},
        cpt=cpt_with_evidence
    )
    print(f"Created node: {node_d} with CPT and evidences")
    try:
        node_d.validate()
        print("Validation passed (expected).")
    except ValueError as e:
        print(f"Validation failed (unexpected): {e}")


    print("\n--- Example 5: Validation Failure (Invalid evidences_states) ---")
    # Node declares evidence B but with different evidence-state names than CPT
    node_e = DBN_Node_intern(
        name="A",
        states=["on", "off"],
        evidences=["B"],
        evidences_states={"B": ["yes", "no"]},  # does not match CPT.evidences_states {"true","false"}
        cpt=cpt_with_evidence
    )
    try:
        node_e.validate()
        print("Validation passed (Unexpected).")
    except ValueError as e:
        print(f"Validation failed (Expected): {e}")
