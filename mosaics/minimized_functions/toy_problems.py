# Contains toy problems for code testing.
from ..random_walk import TrajectoryPoint
from ..data import NUCLEAR_CHARGE


class Diatomic_barrier:
    def __init__(self, possible_nuclear_charges: list):
        """
        Toy problem potential: If you can run a Monte Carlo simulation in chemical space of diatomic molecules with two elements available this minimization function will
        create one global minimum for A-A, one local minimum for B-B, and A-B as a maximum that acts as a transition state for single-replica MC moves.
        possible_nuclear_charges : nuclear charges considered
        """
        self.larger_nuclear_charge = max(possible_nuclear_charges)
        # Mainly used for testing purposes.
        self.call_counter = 0

    def __call__(self, trajectory_point_in: TrajectoryPoint) -> int:
        self.call_counter += 1
        cg = trajectory_point_in.egc.chemgraph
        return self.ncharge_pot(cg) + self.bond_pot(cg)

    def ncharge_pot(self, cg):
        if cg.hatoms[0].ncharge == cg.hatoms[1].ncharge:
            if cg.hatoms[0].ncharge == self.larger_nuclear_charge:
                return 1
            else:
                return 0
        else:
            return 2

    def bond_pot(self, cg):
        return cg.bond_order(0, 1) - 1


class OrderSlide:
    def __init__(
        self,
        possible_nuclear_charges: list or None = None,
        possible_elements: list or None = None,
    ):
        """
        Toy problem potential for minimizing nuclear charges of heavy atoms of molecules.
        """
        self.call_counter = 0
        if possible_nuclear_charges is None:
            possible_nuclear_charges = [NUCLEAR_CHARGE[el] for el in possible_elements]
        used_nuclear_charges = sorted(possible_nuclear_charges)
        self.order_dict = {}
        for i, ncharge in enumerate(used_nuclear_charges):
            self.order_dict[ncharge] = i

    def __call__(self, trajectory_point_in: TrajectoryPoint):
        self.call_counter += 1
        return -sum(
            self.order_dict[ha.ncharge]
            for ha in trajectory_point_in.egc.chemgraph.hatoms
        )


class ZeroFunc:
    """
    Unbiased chemical space exploration.
    """

    def __init__(self):
        self.call_counter = 0

    def __call__(self, trajectory_point_in: TrajectoryPoint):
        self.call_counter += 1
        return 0.0


class NumHAtoms:
    """
    Return number of heavy atoms.
    """

    def __init__(self, intervals=None):
        self.call_counter = 0
        self.intervals = intervals

    def int_output(self, trajectory_point_in):
        cur_nha = trajectory_point_in.egc.num_heavy_atoms()
        if self.intervals is None:
            return cur_nha
        else:
            for interval_id, val_interval in enumerate(self.intervals):
                if isinstance(val_interval, int):
                    if cur_nha == val_interval:
                        return interval_id
                else:
                    if cur_nha in val_interval:
                        return interval_id
            return None

    def __call__(self, trajectory_point_in: TrajectoryPoint):
        self.call_counter += 1
        return self.int_output(trajectory_point_in)


class ChargeSum(ZeroFunc):
    """
    Toy problem potential for minimizing sum of nuclear charges.
    """

    def __call__(self, trajectory_point_in: TrajectoryPoint):
        self.call_counter += 1
        return sum(ha.ncharge for ha in trajectory_point_in.egc.chemgraph.hatoms)
