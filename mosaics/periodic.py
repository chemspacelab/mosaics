from .misc_procedures import InvalidGraphDefinition


class InvalidElement(KeyError, InvalidGraphDefinition):
    pass


class ElementDict(dict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise InvalidElement(
                f"Element not found: {key}, if necessary consider creating it with mosaics.periodic.add_custom_element"
            )


# Possible valence values.
valences_int = ElementDict(
    {
        1: 1,
        2: 0,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 3,
        8: 2,
        9: 1,
        10: 0,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: (3, 5),
        16: (2, 4, 6),
        17: 1,
        18: 0,
        30: 2,
        32: 4,
        35: 1,
        53: 1,
    }
)
period_int = ElementDict(
    {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 3,
        12: 3,
        13: 3,
        14: 3,
        15: 3,
        16: 3,
        17: 3,
        18: 3,
    }
)

# Number of unshared pairs in valence shell.
unshared_pairs = {7: {3: 1}, 8: {2: 2}, 9: {1: 3}, 15: {3: 1}, 16: {2: 2}, 17: {1: 3}}

# Hybridization for a given "effective coordination number".

coord_num_hybrid = {4: "sp3", 3: "sp2", 2: "sp"}

max_ecn = 4

# Numbers of electrons of different type in valence shell.
s_int = ElementDict(
    {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 1,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 2,
        17: 2,
        18: 2,
    }
)

p_int = ElementDict(
    {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
        10: 6,
        11: 0,
        12: 0,
        13: 1,
        14: 2,
        15: 3,
        16: 4,
        17: 5,
        18: 6,
    }
)

# Priority list of charged species.
# Introduced in order charges to C were assigned only if options of fitting valences with adding charges to O and N were exhausted.
charge_feasibility_list = [[5, 7, 8, 16], [], [6]]
available_charges_lists = [{5: -1, 7: 1, 8: -1, 16: 1}, {7: -1}, {6: -1}]
# which valences can be
charged_valences_int = {5: {-1: 4}, 6: {-1: 3}, 7: {1: 4, -1: 2}, 8: {-1: 1}, 16: {1: 3}}


def get_max_charge_feasibility():
    return len(charge_feasibility_list)


def adjust_list_length(adjusted_list, new_max_index, default_element):
    if len(adjusted_list) > new_max_index:
        return
    for _ in range(new_max_index + 1 - len(adjusted_list)):
        adjusted_list.append(default_element)


def add_custom_element(ncharge, valences, available_charges=None, charged_valences=None):
    valences_int[ncharge] = valences
    if available_charges is not None:
        charge_feasibility_index = min(available_charges.keys())
        adjust_list_length(charge_feasibility_list, charge_feasibility_index, [])
        charge_feasibility_list[charge_feasibility_index].append(ncharge)
        for feasibility_index, charges in available_charges.items():
            adjust_list_length(available_charges_lists, feasibility_index, {})
            available_charges_lists[feasibility_index][ncharge] = charges
    if charged_valences is not None:
        charged_valences_int[ncharge] = charged_valences
