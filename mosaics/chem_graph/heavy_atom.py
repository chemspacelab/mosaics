"""
All procedures for handling nodes: heavy atoms with the hydrogens connected to them.
"""

# NOTE For now I am leaving min_valence, max_valence, and next_valence working only with uncharged atoms, because they are only called during RandomWalks, which cannot currently change charges anyway.
from sortedcontainers import SortedList

from ..misc_procedures import InvalidAdjMat, int_atom_checked, str_atom_corr
from ..periodic import (
    available_charges_lists,
    charge_feasibility_list,
    charged_valences_int,
    p_int,
    period_int,
    s_int,
    valences_int,
)


def can_be_charged(atom_id, charge_feasibility=0):
    for cur_charge_list in charge_feasibility_list[:charge_feasibility]:
        if atom_id in cur_charge_list:
            return True
    return False


def avail_val_list(atom_id, charge=0):
    """
    Possible valence values.
    """
    atom_id = int_atom_checked(atom_id)
    if (charge is None) or (charge == 0):
        return valences_int[atom_id]
    else:
        return charged_valences_int[atom_id][charge]


def cut_avail_val_list(atom_id, charge=0, coordination_number=0):
    l = avail_val_list(atom_id, charge=charge)
    if isinstance(l, int):
        if l >= coordination_number:
            return SortedList([l])
        else:
            return SortedList([])
    else:
        l = SortedList(l)
        cut_id = l.bisect(coordination_number)
        if (cut_id != 0) and (l[cut_id - 1] == coordination_number):
            cut_id -= 1
        return l[cut_id:]


def add_avail_charges_valences(
    avail_charges, avail_valences, atom_id, charge, coordination_number
):
    cur_avail_vals = cut_avail_val_list(
        atom_id, charge=charge, coordination_number=coordination_number
    )
    if len(cur_avail_vals) == 0:
        return False
    avail_charges.append(charge)
    avail_valences.append(cur_avail_vals)
    return cur_avail_vals[-1] != coordination_number


def default_valence(atom_id):
    """
    Default valence for a given element.
    """
    val_list = avail_val_list(atom_id)
    if isinstance(val_list, tuple):
        return val_list[0]
    else:
        return val_list


class HeavyAtom:
    def __init__(
        self,
        atom_symbol,
        nhydrogens=0,
        valence=None,
        possible_valences=None,
        charge=0,
        charge_feasibility=0,
        possible_charges=None,
    ):
        """
        Class storing information about a heavy atom and the hydrogens connected to it.
        """
        self.ncharge = int_atom_checked(atom_symbol)
        self.nhydrogens = nhydrogens
        # these attributes can be altered by ChemGraph object depending on the resonance structure
        self.charge = charge
        self.valence = valence
        # this attribute is assigned based on how "desperate" the code gets in terms of assigning charges to create valid resonance structures
        self.charge_feasibility = charge_feasibility
        # list of possible valences and charges that can be assigned for different resonance structures
        self.possible_valences = possible_valences
        self.possible_charges = possible_charges

    def get_available_valences_charges(self, coordination_number, charge_feasibility=0):
        """
        Enumerate all possible charge and valence states that heavy atom can take given a coordination number.

        Returns:
            Tuple (avail_charges, avail_valences, has_extra_valence). avail_charges is list of available charge values, avail_valences is list of lists of valences available for a given charge, has_extra_valences is bool value indicating whether the atom contains an extra valence.
        """
        avail_charges = []
        avail_valences = []

        has_extra_valence = add_avail_charges_valences(
            avail_charges, avail_valences, self.ncharge, 0, coordination_number
        )
        if self.can_be_charged(charge_feasibility=charge_feasibility):
            additional_charges = []
            for avail_charges_dict in available_charges_lists[:charge_feasibility]:
                if self.ncharge not in avail_charges_dict:
                    continue
                cur_add_charges = avail_charges_dict[self.ncharge]
                if isinstance(cur_add_charges, int):
                    additional_charges.append(cur_add_charges)
                else:
                    additional_charges += cur_add_charges
            for charge in additional_charges:
                cur_has_extra_valence = add_avail_charges_valences(
                    avail_charges,
                    avail_valences,
                    self.ncharge,
                    charge,
                    coordination_number,
                )
                has_extra_valence = has_extra_valence or cur_has_extra_valence
        if len(avail_charges) == 0:
            raise InvalidAdjMat
        if has_extra_valence:
            self.valence = None
        else:
            self.valence = avail_valences[0][0]
        if len(avail_charges) == 1:
            self.charge = avail_charges[0]
        else:
            self.charge = None

        return avail_charges, avail_valences, has_extra_valence

    def clear_possibilities(self):
        self.valence = None
        self.charge = None
        self.possible_valences = None
        self.possible_charges = None

    def is_polyvalent(self):
        return isinstance(valences_int[self.ncharge], tuple)

    def can_be_charged(self, charge_feasibility=0):
        return can_be_charged(self.ncharge, charge_feasibility=charge_feasibility)

    def charge_reasonable(self, charge_feasibility=0):
        if self.can_be_charged(charge_feasibility=charge_feasibility):
            for avail_charge_dict in available_charges_lists[:charge_feasibility]:
                if self.ncharge not in avail_charge_dict:
                    continue
                charges = avail_charge_dict[self.ncharge]
                if isinstance(charges, int):
                    if self.charge == charges:
                        return True
                else:
                    if self.charge in charges:
                        return True
            return False
        else:
            return self.charge == 0

    def valence_reasonable(self):
        val_list = self.avail_val_list(charge=self.charge)
        if isinstance(val_list, tuple):
            return self.valence in val_list
        else:
            return self.valence == val_list

    def avail_val_list(self, **kwargs):
        return avail_val_list(self.ncharge, **kwargs)

    def min_valence(self):
        val_list = self.avail_val_list()
        if isinstance(val_list, tuple):
            return val_list[0]
        else:
            return val_list

    def max_valence(self):
        val_list = self.avail_val_list()
        if isinstance(val_list, tuple):
            return val_list[-1]
        else:
            return val_list

    def mincopy(self):
        """
        Not %100 sure whether this should be made __deepcopy__ instead.
        """
        return HeavyAtom(
            atom_symbol=self.ncharge, valence=self.valence, nhydrogens=self.nhydrogens
        )

    def element_name(self):
        return str_atom_corr(self.ncharge)

    # Procedures for ordering.
    def get_comparison_list(self):
        return [self.ncharge, self.nhydrogens]

    def __lt__(self, ha2):
        return self.get_comparison_list() < ha2.get_comparison_list()

    def __gt__(self, ha2):
        return self.get_comparison_list() > ha2.get_comparison_list()

    def __eq__(self, ha2):
        return self.get_comparison_list() == ha2.get_comparison_list()

    # Procedures for printing.
    def __str__(self):
        output = str(self.ncharge)
        if self.nhydrogens != 0:
            output += "#" + str(self.nhydrogens)
        return output

    def __repr__(self):
        return str(self)


# TODO check that the function is not duplicated elsewhere
def next_valence(ha: HeavyAtom, int_step: int = 1, valence_option_id: int or None = None):
    """
    Next valence value.
    """
    # check valences have been initialized
    assert ha.valence is not None
    val_list = ha.avail_val_list()
    if (valence_option_id is not None) and (ha.possible_valences is not None):
        cur_valence = ha.possible_valences[valence_option_id]
    else:
        cur_valence = ha.valence
    cur_val_id = val_list.index(cur_valence)
    new_val_id = cur_val_id + int_step
    if (new_val_id < 0) or (new_val_id >= len(val_list)):
        return None
    else:
        return val_list[new_val_id]


# Functions that should help defining a meaningful distance measure between Heavy_Atom objects. TODO: Still need those?
def hatom_state_coords(ha):
    return [
        period_int[ha.ncharge],
        s_int[ha.ncharge],
        p_int[ha.ncharge],
        ha.valence,
        ha.nhydrogens,
    ]


num_state_coords = {hatom_state_coords: 5}


def str2HeavyAtom(ha_str: str):
    fields = ha_str.split("#")
    return HeavyAtom(int(fields[0]), nhydrogens=int(fields[1]))
