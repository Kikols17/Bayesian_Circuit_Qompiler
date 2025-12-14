from collections import defaultdict
from typing import (
    List,
    Dict,
    Tuple,
    Iterable,
    Optional,
    Set,
)

class CPT_intern:
    """
    Represents a Conditional Probability Table.

    - variable: the query variable name (node)
    - variable_states: optional list specifying the ordered states of the variable
    - evidences: list of evidence/parent variable names
    - table: mapping from tuples of values (variable_value, *evidence_values)
             to a probability (float in [0,1])
    """

    variable: str                               # Query variable name
    variable_states: List[str | bool]           # Ordered list of variable states names (bool for binary states, string otherwise)
    evidences: List[str]                        # Evidence variables' names
    table: Dict[Tuple[str | bool, ...], float]  # CPT table in Dict form

    def __init__(self,
                 variable: str,
                 variable_states: Optional[List[str|bool]] = None,
                 evidences: Optional[List[str]] = None,
                 table: Optional[Dict[Tuple[str|bool, ...], float]] = None
                ) -> None:
        self.variable = variable
        self.variable_states = variable_states
        self.evidences = evidences
        self.table = {}
        if table is not None:
            self.table = table

    def set_table(self,
                  table: Dict[Tuple[str|bool, ...], float]
                 ) -> None:
        """
        Replace the whole table. Keys must be tuples with length == 1 + len(evidences).
        Values must be numbers in [0, 1].
        If variable_states is set, every key's variable value must be one of those states.
        """
        validated: Dict[Tuple[str|bool, ...], float] = {}
        expected_len = 1 + len(self.evidences)
        for k, v in table.items():
            if not isinstance(k, tuple):
                raise TypeError("table keys must be tuples matching (variable, *evidences) order")
            if len(k) != expected_len:
                raise ValueError("table key length does not match number of variable + evidences")
            if not isinstance(v, (int, float)):
                raise TypeError("table values must be numeric (int or float)")
            prob = float(v)
            if prob < 0.0 or prob > 1.0:
                raise ValueError("probabilities must be in [0, 1]")
            if self.variable_states is not None:
                if k[0] not in self.variable_states:
                    raise ValueError(f"variable value {k[0]!r} not in declared variable_states")
            validated[k] = prob
        self.table = validated

    def get_table(self) -> Dict[Tuple[str|bool, ...], float]:
        """Return a shallow copy of the table."""
        return dict(self.table)

    def get_tablesize(self) -> Tuple[int]:
        """Return the size (cols, rows) of the table."""
        # determine variable values (rows)
        if self.variable_states is not None:
            var_vals: List[str|bool] = list(self.variable_states)
        else:
            var_vals = []
        # determine distinct parent assignments (columns minus the first column)
        parent_keys: List[Tuple[str|bool, ...]] = []
        seen_parents: Set[Tuple[str|bool, ...]] = set()
        seen_vars = set(var_vals)

        for key in self.table.keys():
            var = key[0]
            parents = key[1:]
            if var not in seen_vars:
                var_vals.append(var)
                seen_vars.add(var)
            if parents not in seen_parents:
                parent_keys.append(parents)
                seen_parents.add(parents)

        cols = len(parent_keys)
        rows = len(var_vals)
        return (cols, rows)

    def set_probability(self,
                        assignment: Iterable[str|bool],
                        prob: float
                       ) -> None:
        """
        Set probability for a single assignment.
        assignment: iterable of values in the order (variable_value, *evidence_values)
        """
        key = tuple(assignment)
        expected_len = 1 + len(self.evidences)
        if len(key) != expected_len:
            raise ValueError("assignment length does not match number of variable + evidences")
        if not isinstance(prob, (int, float)):
            raise TypeError("probability must be numeric")
        p = float(prob)
        if p < 0.0 or p > 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        if self.variable_states is not None and key[0] not in self.variable_states:
            raise ValueError(f"variable value {key[0]!r} not in declared variable_states")
        self.table[key] = p

    def get_probability(self,
                        query: str|bool,
                        evidence: Iterable[str|bool]
                       ) -> float:
        """Return the probability for the given query | evidence (raises KeyError if missing)."""
        # Normalize evidence: allow a single value (e.g. "summer") to be passed directly,
        # and treat strings/bytes as single values instead of iterables of characters.
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Iterable):
            ev_tuple = (evidence,)
        else:
            ev_tuple = tuple(evidence)

        key = (query,) + ev_tuple
        expected_len = 1 + len(self.evidences)
        if len(key) != expected_len:
            raise ValueError(f"query + assignment length ({len(key)}) does not match number of variable + evidences ({expected_len})")
        return self.table[key]

    def validate(self,
                 tol: float = 1e-8
                ) -> bool:
        """
        Validate table structure and content.

        Raises:
            ValueError, TypeError on any invalidity.
        Checks performed:
        - variable is a non-empty string
        - table keys are tuples of length 1 + len(evidences)
        - table values are numeric and in [0,1]
        - if variable_states is provided, every key's variable value is in variable_states
        - for each evidence assignment, the probabilities over variable values sum to 1 (within tol)
        - if variable_states is provided, for each evidence assignment the set of variable values
          equals the declared variable_states (no missing or extra states)
        Returns True if valid.
        """
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError("variable must be a non-empty string")

        evidences = self.evidences or []
        expected_len = 1 + len(evidences)

        sums: Dict[Tuple[str|bool, ...], float] = defaultdict(float)
        seen_vals: Dict[Tuple[str|bool, ...], Set[str|bool]] = defaultdict(set)

        for k, v in self.table.items():
            if not isinstance(k, tuple):
                raise TypeError(f"table keys must be tuples matching (variable, *evidences) order (\"{k}\" is \"{type(k)}\")")
            if len(k) != expected_len:
                raise ValueError(f"table key length {len(k)} does not match number of variable + evidences ({expected_len})")
            if not isinstance(v, (int, float)):
                raise TypeError(f"table values must be numeric (int or float) (\"{v}\" is \"{type(v)}\")")
            prob = float(v)
            if prob < 0.0 or prob > 1.0:
                raise ValueError(f"probability {prob} out of [0, 1]")
            if self.variable_states is not None and k[0] not in self.variable_states:
                raise ValueError(f"variable value \"{k[0]!r}\" not in declared variable_states ({self.variable_states})")
            evidence_key = k[1:]
            sums[evidence_key] += prob
            seen_vals[evidence_key].add(k[0])

        if not sums:
            raise ValueError("CPT table is empty")

        for evidence_key, total in sums.items():
            if abs(total - 1.0) > tol:
                raise ValueError(f"probabilities for evidences {evidence_key} sum to {total}, not 1.0")
            if self.variable_states is not None:
                seen = seen_vals.get(evidence_key, set())
                if set(self.variable_states) != seen:
                    raise ValueError(f"for evidences {evidence_key} variable states {seen} do not match declared states {set(self.variable_states)}")

    def table_string(self,
                     float_fmt: str = "{:.4g}",
                     missing: str = ""
                    ) -> str:
        """
        Return an ASCII table (string) representing the CPT.

        Columns are evidence assignments (evidences), rows are variable values.
        If multiple evidences are present, each evidence(name(value)) is stacked
        vertically in the column header.
        If variable_states is set it controls the row order; otherwise the order is inferred
        from the keys in self.table.
        """
        # collect variable values (first element) and evidence assignments in encountered order
        if self.variable_states is not None:
            var_vals: List[str|bool] = list(self.variable_states)
        else:
            var_vals = []
        parent_keys: List[Tuple[str|bool, ...]] = []
        seen_vars = set(var_vals)
        seen_parents = set()
        for key in self.table.keys():
            var = key[0]
            parents = key[1:]
            if var not in seen_vars:
                var_vals.append(var)
                seen_vars.add(var)
            if parents not in seen_parents:
                parent_keys.append(parents)
                seen_parents.add(parents)

        # If table empty, return simple representation
        if not var_vals or not parent_keys:
            return f"CPT_intern(variable={self.variable}, evidences={self.evidences}, entries={len(self.table)})"

        # Build header as multiple stacked lines when there are multiple evidences
        parents = self.evidences or []
        header_height = max(1, len(parents))

        # first column header lines (variable label on top, empty below to align)
        first_col_header_lines = [self.variable] + [""] * (header_height - 1)

        # parent column header lines: for each parent assignment, produce one line per evidence
        parent_header_lines: List[List[str]] = []
        for pk in parent_keys:
            if not parents:
                # single-line "prob" header if no evidence names given
                parent_header_lines.append([ "prob" ] + [""] * (header_height - 1))
            else:
                lines = []
                for name, val in zip(parents, pk):
                    # If state is a str, print it, if bool, put 0/1
                    if isinstance(val, bool):
                        lines.append(f"{name}({int(val)})")
                    else:
                        lines.append(f"{name}({val})")
                # if there are fewer values than parents (shouldn't happen) pad
                if len(lines) < header_height:
                    lines += [""] * (header_height - len(lines))
                parent_header_lines.append(lines)

        # build header rows (each row is one of the stacked header lines)
        header_rows: List[List[str]] = []
        for r in range(header_height):
            row = [first_col_header_lines[r]]
            for ph in parent_header_lines:
                row.append(ph[r])
            header_rows.append(row)

        # build row labels and cells
        # If state is a str, print it, if bool, put 0/1
        row_labels: List[str] = []
        for v in var_vals:
            if isinstance(v, bool):
                row_labels.append(f"{self.variable}({int(v)})")
            else:
                row_labels.append(f"{self.variable}({v})")

        body: List[List[str]] = []
        for vv in var_vals:
            row: List[str] = []
            for pk in parent_keys:
                key = (vv,) + tuple(pk)
                if key in self.table:
                    row.append(float_fmt.format(self.table[key]))
                else:
                    row.append(missing)
            body.append(row)

        # determine column widths (including first column)
        cols = 1 + len(parent_keys)
        col_widths: List[int] = [0] * cols
        # first col width from header_lines and row labels
        max_first_header_len = max(len(line) for line in first_col_header_lines)
        col_widths[0] = max(max_first_header_len, *(len(lbl) for lbl in row_labels))
        for j in range(1, cols):
            # header lines for this column are in parent_header_lines[j-1]
            max_header_len = max(len(line) for line in parent_header_lines[j-1])
            col_widths[j] = max(max_header_len, *(len(body[i][j-1]) for i in range(len(body))))

        # build horizontal separator
        def sep_line() -> str:
            parts = ["+"]
            for w in col_widths:
                parts.append("-" * (w + 2))
                parts.append("+")
            return "".join(parts)

        # build a row given list of cell strings
        def build_row(cells: List[str]) -> str:
            parts = ["|"]
            for i, cell in enumerate(cells):
                parts.append(" " + cell.ljust(col_widths[i]) + " ")
                parts.append("|")
            return "".join(parts)

        lines: List[str] = []
        lines.append(sep_line())
        # add each header stacked line
        for hr in header_rows:
            lines.append(build_row(hr))
        lines.append(sep_line())
        for label, row in zip(row_labels, body):
            # first column is the label, remaining are row cells
            lines.append(build_row([label] + row))
            lines.append(sep_line())

        return "\n".join(lines)


    def __repr__(self) -> str:
        return (f"CPT_intern(variable={self.variable!r}, evidences={self.evidences}, "
                f"states={self.variable_states}, entries={len(self.table)})")


if __name__ == "__main__":
    # To run this example: python -m src.base.CPT_intern

    print("--- Example 1: Simple Marginal Probability (No Parents) ---")
    # A simple coin toss
    cpt_coin = CPT_intern(
        variable="Coin",
        variable_states=["Heads", "Tails"],
        evidences=[]
    )
    cpt_coin.set_table({
        ("Heads",): 0.5,
        ("Tails",): 0.5,
    })
    print(f"Created CPT for {cpt_coin.variable}")
    print(cpt_coin.table_string())
    print("Table size:", cpt_coin.get_tablesize())
    try:
        cpt_coin.validate()
        print("Validation passed.")
    except ValueError as e:
        print(f"Validation failed (NOT SUPOSED TO HAPPEN): {e}")


    print("\n--- Example 2: Conditional Probability (One Parent) ---")
    # Grass Wet depends on Rain
    cpt_grass = CPT_intern(
        variable="GrassWet",
        variable_states=[True, False],
        evidences=["Rain"]
    )
    # P(GrassWet | Rain)
    cpt_grass.set_table({
        (True, True): 0.9,      # P(Wet=yes | Rain=yes)
        (False, True): 0.1,     # P(Wet=no  | Rain=yes)
        (True, False): 0.2,     # P(Wet=yes | Rain=no) - maybe sprinkler?
        (False, False): 0.8,    # P(Wet=no  | Rain=no)
    })
    print(cpt_grass.table_string())
    print(f"P(GrassWet=yes | Rain=yes) = {cpt_grass.get_probability(True, True)}")
    print("Table size:", cpt_grass.get_tablesize())


    print("\n--- Example 3: Conditional Probability (One Parent with Non-Binary States) ---")
    # GrassWet depends on Weather which has three states: sunny, cloudy, rainy
    cpt_grass = CPT_intern(
        variable="GrassWet",
        variable_states=[True, False],
        evidences=["Weather"]
    )
    # P(GrassWet | Weather)
    cpt_grass.set_table({
        (True,  "sunny"):  0.01,
        (False, "sunny"):  0.99,
        (True,  "cloudy"): 0.30,
        (False, "cloudy"): 0.70,
        (True,  "rainy"):  0.95,
        (False, "rainy"): 0.05,
    })
    print(cpt_grass.table_string())
    print(f"P(GrassWet=True | Weather='rainy') = {cpt_grass.get_probability(True, 'rainy')}")
    print("Table size:", cpt_grass.get_tablesize())


    print("\n--- Example 4: Multiple Parents & Incremental Setup ---")
    # Alarm depends on Burglary and Earthquake
    cpt_alarm = CPT_intern(
        variable="Alarm",
        variable_states=["ring", "silent"],
        evidences=["Burglary", "Earthquake"]
    )

    # Setting probabilities one by one
    # P(Alarm | Burglary, Earthquake)
    cpt_alarm.set_probability(("ring",   "yes", "yes"), 0.95)
    cpt_alarm.set_probability(("silent", "yes", "yes"), 0.05)

    cpt_alarm.set_probability(("ring",   "yes", "no"),  0.94)
    cpt_alarm.set_probability(("silent", "yes", "no"),  0.06)

    cpt_alarm.set_probability(("ring",   "no",  "yes"), 0.29)
    cpt_alarm.set_probability(("silent", "no",  "yes"), 0.71)

    cpt_alarm.set_probability(("ring",   "no",  "no"),  0.001)
    cpt_alarm.set_probability(("silent", "no",  "no"),  0.999)

    print(cpt_alarm.table_string())
    print("Table size:", cpt_alarm.get_tablesize())

    try:
        cpt_alarm.validate()
        print("Validation passed.")
    except ValueError as e:
        print(f"Validation failed (NOT SUPOSED TO HAPPEN): {e}")



    print("\n--- Example 5: Validation Errors ---")
    cpt_broken = CPT_intern(
        variable="Broken",
        variable_states=["yes", "no"],
        evidences=[]
    )
    # Probabilities sum to 1.1
    cpt_broken.set_table({
        ("yes",): 0.6,
        ("no",): 0.5
    })
    print("Created invalid table (sum > 1)")
    print(cpt_broken.table_string())
    print("Table size:", cpt_broken.get_tablesize())
    try:
        cpt_broken.validate()
        print("Validation passed (NOT SUPOSED TO HAPPEN).")
    except ValueError as e:
        print(f"Validation failed (SUPOSED TO HAPPEN): {e}")
