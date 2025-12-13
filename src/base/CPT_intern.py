from collections import defaultdict
from typing import (
    List,
    Dict,
    Tuple,
    Any,
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
                 variable_states: Optional[List[Any]] = None,
                 evidences: Optional[List[str]] = None,
                 table: Optional[Dict[Tuple[Any, ...], float]] = None
                ) -> None:
        self.variable = variable
        self.variable_states = variable_states
        self.evidences = evidences
        self.table = {}
        if table is not None:
            self.table = table

    def set_table(self,
                  table: Dict[Tuple[Any, ...], float]
                 ) -> None:
        """
        Replace the whole table. Keys must be tuples with length == 1 + len(evidences).
        Values must be numbers in [0, 1].
        If variable_states is set, every key's variable value must be one of those states.
        """
        validated: Dict[Tuple[Any, ...], float] = {}
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

    def get_table(self) -> Dict[Tuple[Any, ...], float]:
        """Return a shallow copy of the table."""
        return dict(self.table)

    def set_probability(self,
                        assignment: Iterable[Any],
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

        sums: Dict[Tuple[Any, ...], float] = defaultdict(float)
        seen_vals: Dict[Tuple[Any, ...], Set[Any]] = defaultdict(set)

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
            var_vals: List[Any] = list(self.variable_states)
        else:
            var_vals = []
        parent_keys: List[Tuple[Any, ...]] = []
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
        row_labels = [f"{self.variable}({v})" for v in var_vals]
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
    # Examples exercising all public functions (and a couple internals).

    # Example 1: variable with one parent (original example)
    cpt = CPT_intern(variable="weather", evidences=["season"],
                     variable_states=["sunny", "cloudy", "rainy"])
    table = {
        ("sunny", "summer"): 0.7,
        ("cloudy", "summer"): 0.2,
        ("rainy", "summer"): 0.1,
        ("sunny", "winter"): 0.2,
        ("cloudy", "winter"): 0.3,
        ("rainy", "winter"): 0.5,
    }
    # set_table -> validates and stores
    cpt.set_table(table)
    print("Initial CPT repr:", repr(cpt))
    # validate -> True
    print("Validate:", cpt.validate())
    # table_string -> ASCII representation
    print("\nASCII table (weather):")
    print(cpt.table_string(float_fmt="{:.2f}"))

    # get_probability -> direct lookup
    p = cpt.get_probability("sunny", ("summer"))
    print("P(weather=sunny | season=summer) =", p)

    # entries / get_table -> get a copy
    entries_copy = cpt.get_table()
    print("Entries size:", len(entries_copy))

    # Example 2: variable with no evidences (simple marginal)
    coin = CPT_intern(variable="coin", evidences=[], variable_states=["H", "T"])
    # set_table with single-element keys
    coin.set_table({
        ("H",): 0.5,
        ("T",): 0.5,
    })
    print("\nCoin CPT repr:", repr(coin))
    print("Coin validate:", coin.validate())
    print("Coin ASCII:")
    print(coin.table_string(float_fmt="{:.1f}"))

    # Example 3: using set_probability and get_table for a multi-parent node
    lights = CPT_intern(variable="light", evidences=["switch1", "switch2"], variable_states=["on", "off"])
    # set individual probabilities (for one evidence assignment)
    lights.set_probability(("on", "up", "up"), 1.0)
    lights.set_probability(("off", "up", "up"), 0.0)
    # for another evidence assignment
    lights.set_probability(("on", "down", "down"), 0.0)
    lights.set_probability(("off", "down", "down"), 1.0)
    print("\nLights entries (partial):", lights.get_table())
    print(lights.table_string())
    # get_probability for a known assignment
    print("P(light=on | switch1=up, switch2=up) =", lights.get_probability("on", ("up", "up")))
    # validate should be False because not all evidence assignments present for both states
    print("Lights validate (should be True):", lights.validate())

    # Example 4: change variable_states after creation
    temp = CPT_intern("temp", evidences=["day", "time"])
    # initially no variable_states; set probabilities for both times (day/night) for monday and tuesday
    temp.set_table({
        ("hot", "monday", "day"): 0.6,
        ("cold", "monday", "day"): 0.4,
        ("hot", "monday", "night"): 0.3,
        ("cold", "monday", "night"): 0.7,
        ("hot", "tuesday", "day"): 0.8,
        ("cold", "tuesday", "day"): 0.2,
        ("hot", "tuesday", "night"): 0.4,
        ("cold", "tuesday", "night"): 0.6,
    })
    print("\nTemp before states repr:", repr(temp))
    # declare states (order matters for printing/validation)
    print("Temp validate (should be False, missing variable_states):", temp.validate())
    temp.variable_states = ["cold", "hot"]
    print(temp.table)
    print("Temp after setting variable_states repr:", repr(temp))
    print("Temp validate (should be True):", temp.validate())
    print(temp.table_string(float_fmt="{:.1f}"))
