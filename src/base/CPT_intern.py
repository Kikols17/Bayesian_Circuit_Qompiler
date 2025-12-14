from collections import defaultdict
from typing import (
    List,
    Dict,
    Tuple,
    Iterable,
    Optional,
    Set,
)
from itertools import product

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
    variable_states: List[str|bool]             # Ordered list of variable states names (bool for binary states, string otherwise)
    evidences: List[str]                        # Evidence variables' names
    evidences_states: Dict[str, List[str|bool]] # Dictionary that associates evidences' variable names to ordered lists of state names
    table: Dict[Tuple[str | bool, ...], float]  # CPT table in Dict form

    def __init__(self,
                 variable: str,
                 variable_states: Optional[List[str|bool]] = None,
                 evidences: Optional[List[str]] = None,
                 evidences_states: Dict[str, List[str | bool]] = None,
                 table: Optional[Dict[Tuple[str|bool, ...], float]] = None
                ) -> None:
        self.variable = variable
        self.variable_states = variable_states
        self.evidences = evidences
        self.evidences_states = evidences_states
        self.table = {}
        if table is not None:
            self.set_table(table=table)

    def get_evidence_comb(self) -> Tuple[List[Tuple[str|bool, ...]], List[Dict[str, str|bool]]]:
        """
        Return the complete list of all possible evidence-state combinations.

        Returns:
            (combos, mappings)
            - combos: list of tuples of evidence values in the same order as self.evidences
              (for no evidences returns [()]).
            - mappings: list of dicts mapping evidence_name -> value for each combo.

        Raises:
            ValueError if evidences_states is missing for any evidence name.
        """
        parents = self.evidences or []
        if not parents:
            return [()], [{}]

        if self.evidences_states is None:
            raise ValueError("evidences_states must be provided to compute all evidence combinations")

        lists = []
        for name in parents:
            states = self.evidences_states.get(name)
            if states is None:
                raise ValueError(f"no states declared for evidence {name!r} in evidences_states")
            lists.append(list(states))

        combos = [tuple(p) for p in product(*lists)]
        mappings: List[Dict[str, str|bool]] = []
        for combo in combos:
            mapping = {name: val for name, val in zip(parents, combo)}
            mappings.append(mapping)
        return combos, mappings

    def set_table(self,
                  table: Dict[Tuple[str|bool, ...], float]
                 ) -> None:
        """
        Replace the whole table. Keys must be tuples with length == 1 + len(evidences).
        Values must be numbers in [0, 1].
        If variable_states is set, every key's variable value must be one of those states.
        If evidences_states is set, every evidence value in each key must be one of the declared states
        for that evidence (evidences_states is expected to be a dict mapping evidence name -> list of states).
        Additionally, this enforces that the table contains exactly the Cartesian product of
        variable_states x evidence combinations (no missing or extra entries).
        """
        expected_len = 1 + len(self.evidences or [])

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

            # check variable state legality
            if self.variable_states is not None:
                if k[0] not in self.variable_states:
                    raise ValueError(f"variable value {k[0]!r} not in declared variable_states")

            # check evidences states legality if provided
            if self.evidences_states is not None:
                parents = k[1:]
                for idx, val in enumerate(parents):
                    # ensure we have an evidence name at this position
                    try:
                        evidence_name = (self.evidences or [])[idx]
                    except IndexError:
                        # should not happen because of earlier length check
                        raise ValueError("mismatch between evidences and table key length")
                    allowed = self.evidences_states.get(evidence_name)
                    if allowed is None:
                        raise ValueError(f"no states declared for evidence {evidence_name!r} in evidences_states")
                    if val not in allowed:
                        raise ValueError(f"evidence value {val!r} for '{evidence_name}' not in declared states {allowed}")

        # enforce completeness: table must contain every variable state for every evidence combination
        if self.variable_states is None:
            raise ValueError("variable_states must be provided to ensure completeness of the CPT")

        # compute expected parent combinations
        combos, _ = self.get_evidence_comb()
        expected_keys = set()
        for parent in combos:
            for var_state in self.variable_states:
                expected_keys.add((var_state,) + tuple(parent))

        table_keys = set(table.keys())
        if table_keys != expected_keys:
            missing = expected_keys - table_keys
            extra = table_keys - expected_keys
            msgs = []
            if missing:
                msgs.append(f"missing entries for combinations: {missing}")
            if extra:
                msgs.append(f"extra/unexpected entries: {extra}")
            raise ValueError("table does not contain exactly the full set of variable states x evidence combinations: " + "; ".join(msgs))

        self.table = table

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
        Note: incremental setting is allowed; completeness is enforced by set_table/validate.
        """
        key = tuple(assignment)
        expected_len = 1 + len(self.evidences or [])
        if len(key) != expected_len:
            raise ValueError("assignment length does not match number of variable + evidences")
        if not isinstance(prob, (int, float)):
            raise TypeError("probability must be numeric")
        p = float(prob)
        if p < 0.0 or p > 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        if self.variable_states is not None and key[0] not in self.variable_states:
            raise ValueError(f"variable value {key[0]!r} not in declared variable_states")
        # evidences states legality check if present
        if self.evidences_states is not None:
            parents = key[1:]
            for idx, val in enumerate(parents):
                try:
                    evidence_name = (self.evidences or [])[idx]
                except IndexError:
                    raise ValueError("mismatch between evidences and assignment length")
                allowed = self.evidences_states.get(evidence_name)
                if allowed is None:
                    raise ValueError(f"no states declared for evidence {evidence_name!r} in evidences_states")
                if val not in allowed:
                    raise ValueError(f"evidence value {val!r} for '{evidence_name}' not in declared states {allowed}")

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
        expected_len = 1 + len(self.evidences or [])
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
        - ensures the table covers every evidence combination (if evidences_states provided)
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

            # check variable state legality
            if self.variable_states is not None:
                if k[0] not in self.variable_states:
                    raise ValueError(f"variable value {k[0]!r} not in declared variable_states")

            # check evidences states legality if provided
            if self.evidences_states is not None:
                parents = k[1:]
                for idx, val in enumerate(parents):
                    # ensure we have an evidence name at this position
                    try:
                        evidence_name = self.evidences[idx]
                    except IndexError:
                        # should not happen because of earlier length check
                        raise ValueError("mismatch between evidences and table key length")
                    allowed = self.evidences_states.get(evidence_name)
                    if allowed is None:
                        raise ValueError(f"no states declared for evidence {evidence_name!r} in evidences_states")
                    if val not in allowed:
                        raise ValueError(f"evidence value {val!r} for '{evidence_name}' not in declared states {allowed}")

        if not sums:
            raise ValueError("CPT table is empty")

        # if evidences_states provided, ensure sums keys cover all combinations
        if self.evidences_states is not None:
            combos, _ = self.get_evidence_comb()
            seen_combos = set(sums.keys())
            expected_combos = set(combos)
            if seen_combos != expected_combos:
                missing = expected_combos - seen_combos
                extra = seen_combos - expected_combos
                msgs = []
                if missing:
                    msgs.append(f"missing evidence combinations: {missing}")
                if extra:
                    msgs.append(f"unexpected evidence combinations: {extra}")
                raise ValueError("evidence combinations in table do not match evidences_states: " + "; ".join(msgs))

        for evidence_key, total in sums.items():
            if abs(total - 1.0) > tol:
                raise ValueError(f"probabilities for evidences {evidence_key} sum to {total}, not 1.0")
            if self.variable_states is not None:
                seen = seen_vals.get(evidence_key, set())
                if set(self.variable_states) != seen:
                    raise ValueError(f"for evidences {evidence_key} variable states {seen} do not match declared states {set(self.variable_states)}")

        return True

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
        # no evidences_states needed because no evidences
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
    # Grass Wet depends on Rain (boolean)
    cpt_grass = CPT_intern(
        variable="GrassWet",
        variable_states=[True, False],
        evidences=["Rain"],
        evidences_states={"Rain": [True, False]}
    )
    # P(GrassWet | Rain)
    cpt_grass.set_table({
        (True, True): 0.9,      # P(Wet=yes | Rain=yes)
        (False, True): 0.1,     # P(Wet=no  | Rain=yes)
        (True, False): 0.2,     # P(Wet=yes | Rain=no)
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
        evidences=["Weather"],
        evidences_states={"Weather": ["sunny", "cloudy", "rainy"]}
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
    # Alarm depends on Burglary and Earthquake (both yes/no)
    cpt_alarm = CPT_intern(
        variable="Alarm",
        variable_states=["ring", "silent"],
        evidences=["Burglary", "Earthquake"],
        evidences_states={
            "Burglary": ["yes", "no"],
            "Earthquake": ["yes", "no"]
        }
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


    print("\n--- Example 6: Failure when evidences_states missing an evidence entry ---")
    # intenionally provide evidences_states missing "Earthquake"
    cpt_fail_missing_state_map = CPT_intern(
        variable="AlarmFail",
        variable_states=["ring", "silent"],
        evidences=["Burglary", "Earthquake"],
        evidences_states={ "Burglary": ["yes", "no"] }  # missing Earthquake
    )
    try:
        cpt_fail_missing_state_map.set_table({
            ("ring", "yes", "yes"): 0.95,
            ("silent", "yes", "yes"): 0.05,
            ("ring", "yes", "no"): 0.94,
            ("silent", "yes", "no"): 0.06,
            ("ring", "no", "yes"): 0.29,
            ("silent", "no", "yes"): 0.71,
            ("ring", "no", "no"): 0.001,
            ("silent", "no", "no"): 0.999,
        })
        print("ERROR: expected set_table to fail due to missing evidences_states entry (NOT HAPPENED)")
    except ValueError as e:
        print(f"Expected failure occurred: {e}")


    print("\n--- Example 7: Failure when an evidence value is not in declared evidence states ---")
    cpt_fail_bad_value = CPT_intern(
        variable="AlarmFail2",
        variable_states=["ring", "silent"],
        evidences=["Burglary", "Earthquake"],
        evidences_states={
            "Burglary": ["yes", "no"],
            "Earthquake": ["yes", "no"]
        }
    )
    try:
        # use an evidence value "maybe" which is not declared for Earthquake
        cpt_fail_bad_value.set_table({
            ("ring", "yes", "maybe"): 0.5,
            ("silent", "yes", "maybe"): 0.5,
        })
        print("ERROR: expected set_table to fail due to invalid evidence value (NOT HAPPENED)")
    except ValueError as e:
        print(f"Expected failure occurred: {e}")


    N = 100
    print(f"\n--- Example 8: Speed test of CPT with {N}x{N} ---")
    import time
    import random
    var_states = [f"s{i}" for i in range(N)]
    parent_states = [f"p{j}" for j in range(N)]

    print(f"Creating a {N}x{N} CPT (rows=variable states, cols=parent states) -> total entries = {N*N}")

    # Build table dict (uniform distribution per parent state)
    t0 = time.perf_counter()
    table = {}
    pval = 1.0 / N
    for p in parent_states:
        for s in var_states:
            table[(s, p)] = pval
    t_build = time.perf_counter() - t0

    # Create CPT and set the whole table
    t0 = time.perf_counter()
    cpt_large = CPT_intern(
        variable="LargeVar",
        variable_states=var_states,
        evidences=["Parent"],
        evidences_states={"Parent": parent_states}
    )
    cpt_large.set_table(table)
    t_set = time.perf_counter() - t0

    # Validate the CPT
    t0 = time.perf_counter()
    cpt_large.validate()
    t_validate = time.perf_counter() - t0

    # get_tablesize
    t0 = time.perf_counter()
    size = cpt_large.get_tablesize()
    t_getsize = time.perf_counter() - t0

    # time many get_probability calls
    t0 = time.perf_counter()
    for _ in range(1000):
        q = random.choice(var_states)
        e = random.choice(parent_states)
        _ = cpt_large.get_probability(q, e)
    t_queries = time.perf_counter() - t0

    # time incremental set_probability (building a CPT entry-by-entry)
    t0 = time.perf_counter()
    cpt_inc = CPT_intern(
        variable="LargeVar_inc",
        variable_states=var_states,
        evidences=["Parent"],
        evidences_states={"Parent": parent_states}
    )
    for (k, v) in table.items():
        cpt_inc.set_probability(k, v)
    t_inc_set = time.perf_counter() - t0

    print(f"Built table dict: entries={len(table)} time={t_build:.4f}s")
    print(f"set_table: time={t_set:.4f}s")
    print(f"validate: time={t_validate:.4f}s")
    print(f"get_tablesize: {size} time={t_getsize:.6f}s")
    print(f"1000 get_probability calls: time={t_queries:.4f}s")
    print(f"Incremental set_probability of {len(table)} entries: time={t_inc_set:.4f}s")


    # Additional tests/examples: evidence_comb demonstrations and expected failures
    print("\n--- Example 9: get_evidence_comb with no evidences ---")
    cpt_no_parents = CPT_intern(variable="A", variable_states=["a", "b"], evidences=[])
    combos, maps = cpt_no_parents.get_evidence_comb()
    print("combos:", combos)
    print("maps:", maps)

    print("\n--- Example 10: get_evidence_comb with multiple evidences ---")
    cpt_multi = CPT_intern(
        variable="X",
        variable_states=["x1", "x2"],
        evidences=["P", "Q"],
        evidences_states={"P": [1, 2], "Q": ["r", "s"]}
    )
    combos, maps = cpt_multi.get_evidence_comb()
    print("expected number of combinations:", len([1,2]) * len(["r","s"]))
    print("combos:", combos)
    print("first 3 mappings:", maps[:3])

    print("\n--- Example 10: get_evidence_comb with multiple evidences ---")
    cpt_multi = CPT_intern(
        variable="X",
        variable_states=["x1", "x2"],
        evidences=["P", "Q"],
        evidences_states={"P": [1, 2], "Q": ["r", "s"]}
    )
    combos, maps = cpt_multi.get_evidence_comb()
    print("expected number of combinations:", len([1,2]) * len(["r","s"]))
    print("combos:", combos)
    print("first 3 mappings:", maps[:3])


    print("\n--- Example 4: Multiple Parents Failure to account for all state combinations ---")
    # Alarm depends on Burglary and Earthquake (both yes/no)
    cpt_alarm = CPT_intern(
        variable="Alarm",
        variable_states=["ring", "silent"],
        evidences=["Burglary", "Earthquake"],
        evidences_states={
            "Burglary": ["yes", "no"],
            "Earthquake": ["yes", "no"]
        }
    )

    # Setting probabilities one by one
    # P(Alarm | Burglary, Earthquake)
    cpt_alarm.set_probability(("ring",   "yes", "yes"), 0.95)
    cpt_alarm.set_probability(("silent", "yes", "yes"), 0.05)

    cpt_alarm.set_probability(("ring",   "yes", "no"),  0.94)
    cpt_alarm.set_probability(("silent", "yes", "no"),  0.06)

    cpt_alarm.set_probability(("ring",   "no",  "yes"), 0.29)
    cpt_alarm.set_probability(("silent", "no",  "yes"), 0.71)

    #cpt_alarm.set_probability(("ring",   "no",  "no"),  0.001)
    #cpt_alarm.set_probability(("silent", "no",  "no"),  0.999)

    print(cpt_alarm.table_string())
    print("(Missing evidence combinations ('no', 'no'))")
    print("Table size:", cpt_alarm.get_tablesize())

    try:
        cpt_alarm.validate()
        print("Validation passed. (NOT SUPOSED TO HAPPEN)")
    except ValueError as e:
        print(f"Validation failed (Expected): {e}")