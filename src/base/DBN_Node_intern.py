from .CPT_intern import CPT_intern

from typing import (
    List,
    Optional
)


class DBN_Node_intern:
    """Internal representation of a Discrete Bayesian Network's Node"""
    name: str                           # Node name
    states: List[str | bool]            # Ordered list of states' names (bool for binary states, string otherwise)
    cpt = CPT_intern                    # CPT object

    def __init__(self,
                 name: Optional[str] = None,
                 states: Optional[List[str]] = None,
                 cpt: Optional[CPT_intern] = None
                ) -> None:
        self.name = name
        self.states = states
        self.cpt = cpt

    def validate(self,
             tol: float = 1e-8
            ):
        # Ensure CPT and states are set before validation
        if self.cpt is None:
            raise ValueError("CPT is not set for this node")
        if self.states is None:
            raise ValueError("States are not set for this node")

        # Raise the exception upstream
        try:
            self.cpt.validate(tol=tol)
        except Exception as e:
            raise ValueError(f"CPT validation failed: {e}") from e

        # Checks if self.states == self.cpt.variable_states
        if getattr(self.cpt, "variable_states", None) != self.states:
            raise ValueError(
                f"Node states {self.states} do not match CPT.variable_states {getattr(self.cpt, 'variable_states', None)}"
            )

    def __repr__(self) -> str:
        if self.cpt is not None:
            return f"DBN_Node_intern(name={self.name}, states={self.states}, cpt_size{self.cpt.get_tablesize()})"
        return f"DBN_Node_intern(name={self.name}, states={self.states})"


if __name__ == "__main__":
    # To run this example: python -m src.base.DBN_Node_intern
    print("--- Example 1: Simple Node Creation ---")
    node_a = DBN_Node_intern(name="A", states=["on", "off"])
    print(f"Created node: {node_a}")


    print("\n--- Example 2: Assign CPT to Node ---")
    # Create a CPT for A
    cpt_a = CPT_intern(
        variable="A",
        variable_states=["on", "off"],
        evidences=[],
        table={("on",): 0.7, ("off",): 0.3}
    )
    node_a.cpt = cpt_a
    print(f"Assigned CPT to node {node_a.name}")
    try:
        node_a.validate()
        print("Validation passed.")
    except ValueError as e:
        print(f"Validation failed: {e}")


    print("\n--- Example 2: Full Node Creation (with CPT) ---")
    node_b = DBN_Node_intern(name="A",
                             states=["on", "off"],
                             cpt=CPT_intern(
                                variable="A",
                                variable_states=["on", "off"],
                                evidences=[],
                                table={("on",): 0.7, ("off",): 0.3}
                             )
                            )
    print(f"Created node: {node_b} with CPT")
    print(f"{node_b.cpt.table_string()}")
    try:
        node_b.validate()
        print("Validation passed.")
    except ValueError as e:
        print(f"Validation failed: {e}")




    print("\n--- Example 3: Validation Failure (State Mismatch) ---")
    node_c = DBN_Node_intern(name="C", states=["high", "low"])
    # CPT has different states for B
    cpt_b = CPT_intern(
        variable="C",
        variable_states=["yes", "no"],
        evidences=[],
        table={("yes",): 0.5, ("no",): 0.5}
    )
    node_c.cpt = cpt_b

    try:
        node_c.validate()
        print("Validation passed (Unexpected).")
    except ValueError as e:
        print(f"Validation failed (Expected): {e}")
