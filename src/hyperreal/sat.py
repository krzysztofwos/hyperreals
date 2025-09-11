"""SAT solver for constraint satisfaction."""

from typing import Dict, Iterable, List, Optional


class SAT:
    """Simple DPLL SAT solver."""

    def __init__(self) -> None:
        self.next_var = 1
        self.var_to_name: Dict[int, str] = {}
        self.name_to_var: Dict[str, int] = {}
        self.cnf: List[List[int]] = []
        # Statistics
        self.solves = 0
        self.unit_props = 0
        self.decisions = 0

    def new_var(self, name: str) -> int:
        """Create or retrieve a variable by name."""
        if name in self.name_to_var:
            return self.name_to_var[name]
        v = self.next_var
        self.next_var += 1
        self.name_to_var[name] = v
        self.var_to_name[v] = name
        return v

    def register_var(self, name: str, var_id: int) -> None:
        """Register an existing variable ID (for deserialization)."""
        self.name_to_var[name] = var_id
        self.var_to_name[var_id] = name
        if var_id >= self.next_var:
            self.next_var = var_id + 1

    def lit(self, name: str, polarity: bool = True) -> int:
        """Create a literal from a variable name."""
        v = self.new_var(name)
        return v if polarity else -v

    def add_clause(self, clause: Iterable[int]) -> None:
        """Add a clause to the CNF formula."""
        self.cnf.append(list(clause))

    def with_unit(self, lit: int) -> "SAT":
        """Create a copy with an additional unit clause."""
        s = SAT()
        s.next_var = self.next_var
        s.var_to_name = dict(self.var_to_name)
        s.name_to_var = dict(self.name_to_var)
        s.cnf = [list(c) for c in self.cnf]
        # Copy statistics
        s.solves = self.solves
        s.unit_props = self.unit_props
        s.decisions = self.decisions
        s.add_clause([lit])
        return s

    def export_cnf(self) -> List[List[int]]:
        """Export the CNF formula."""
        return [list(c) for c in self.cnf]

    def _unit_propagate(
        self, cnf: List[List[int]], assign: Dict[int, bool]
    ) -> tuple[bool, Dict[int, bool]]:
        """Perform unit propagation."""
        changed = True
        while changed:
            changed = False
            for clause in cnf:
                val = None
                unassigned = []
                for lit in clause:
                    v = abs(lit)
                    pol = lit > 0
                    if v in assign:
                        if assign[v] == pol:
                            val = True
                            break
                    else:
                        unassigned.append((v, pol))
                if val is True:
                    continue
                if not unassigned:
                    return False, assign
                if len(unassigned) == 1:
                    v, pol = unassigned[0]
                    if v in assign:
                        if assign[v] != pol:
                            return False, assign
                    else:
                        assign[v] = pol
                        self.unit_props += 1
                        changed = True
        return True, assign

    def _choose_unassigned(self, assign: Dict[int, bool]) -> Optional[int]:
        """Choose an unassigned variable."""
        for v in range(1, self.next_var):
            if v not in assign:
                return v
        return None

    def _dpll(
        self, cnf: List[List[int]], assign: Dict[int, bool]
    ) -> Optional[Dict[int, bool]]:
        """DPLL algorithm for SAT solving."""
        ok, assign = self._unit_propagate(cnf, assign)
        if not ok:
            return None
        # Check if satisfied
        for clause in cnf:
            if not any(
                (abs(lit) in assign and assign[abs(lit)] == (lit > 0)) for lit in clause
            ):
                break
        else:
            return assign
        v = self._choose_unassigned(assign)
        if v is None:
            return assign
        self.decisions += 1
        for pol in (True, False):
            assign2 = dict(assign)
            assign2[v] = pol
            res = self._dpll(cnf, assign2)
            if res is not None:
                return res
        return None

    def solve(self) -> Optional[Dict[int, bool]]:
        """Solve the SAT instance."""
        self.solves += 1
        return self._dpll(self.cnf, {})
