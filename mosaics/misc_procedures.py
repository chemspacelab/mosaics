# Several auxiliary functions that appear everywhere.
import copy
import os
import random
from itertools import groupby

import numpy as np

from .data import NUCLEAR_CHARGE


class InvalidAdjMat(Exception):
    pass


# Related to verbosity.

VERBOSITY_MUTED = 0
VERBOSITY_FULL = 1

VERBOSITY_OPTIONS = [VERBOSITY_MUTED, VERBOSITY_FULL]

VERBOSITY = VERBOSITY_MUTED


def set_verbosity(verbosity_value: int):
    assert verbosity_value in VERBOSITY_OPTIONS
    global VERBOSITY
    VERBOSITY = verbosity_value


# End of procedures related to verbosity.


def canonical_atomtype(atomtype):
    return atomtype[0].upper() + atomtype[1:].lower()


def checked_environ_val(
    environ_name: str, expected_answer=None, default_answer=None, var_class=int
):
    """
    Returns os.environ while checking for exceptions.
    """
    if expected_answer is None:
        try:
            args = (os.environ[environ_name],)
        except LookupError:
            if default_answer is None:
                args = tuple()
            else:
                args = (default_answer,)
        return var_class(*args)
    else:
        return expected_answer


default_parallel_backend = "multiprocessing"

num_procs_name = "MOSAICS_NUM_PROCS"


def default_num_procs(num_procs=None):
    return checked_environ_val(num_procs_name, expected_answer=num_procs, default_answer=1)


# Sorting-related.
#
def sorted_by_membership(membership_vector, l=None):
    """
    Sort a list into several lists by membership. Entries with negative membership values are ignored.
    """
    if l is None:
        l = list(range(len(membership_vector)))
    n = max(membership_vector)
    output = [[] for _ in range(n + 1)]
    for val, m in zip(l, membership_vector):
        if m >= 0:
            output[m].append(val)
    return output


# Sorted a tuple either by its value or by value of ordering tuple.
def sorted_tuple(*orig_tuple, ordering_tuple=None):
    if ordering_tuple is None:
        return tuple(sorted(orig_tuple))
    else:
        temp_list = [(i, io) for i, io in zip(orig_tuple, ordering_tuple)]
        temp_list.sort(key=lambda x: x[1])
        return tuple([i for (i, io) in temp_list])


# Sort several tuples.
def sorted_tuples(*orig_tuples):
    output = []
    for orig_tuple in orig_tuples:
        output.append(sorted_tuple(*orig_tuple))
    return sorted(output)


# Dictionnary which is inverse to NUCLEAR_CHARGE in .data
ELEMENTS = None


def str_atom_corr(ncharge):
    """ """
    global ELEMENTS
    if ELEMENTS is None:
        ELEMENTS = {}
        for cur_el, cur_ncharge in NUCLEAR_CHARGE.items():
            ELEMENTS[cur_ncharge] = cur_el
    return ELEMENTS[ncharge]


def int_atom(element):
    """
    Convert string representation of an element to nuclear charge.
    """
    return NUCLEAR_CHARGE[canonical_atomtype(element)]


def int_atom_checked(atom_id):
    """
    Check that input is integer; if string convert to nuclear charge.
    """
    if isinstance(atom_id, str):
        return int_atom(atom_id)
    else:
        return atom_id


# Auxiliary class used for smooth cutoff of positive weights.


class weighted_array(list):
    def normalize_rhos(self, normalization_constant=None):
        if normalization_constant is None:
            normalization_constant = sum(el.rho for el in self)
        for i in range(len(self)):
            self[i].rho /= normalization_constant

    def sort_rhos(self):
        self.sort(key=lambda x: x.rho, reverse=True)

    def normalize_sort_rhos(self):
        self.normalize_rhos()
        self.sort_rhos()

    def cutoff_minor_weights(self, remaining_rho=None):
        if (remaining_rho is not None) and (len(self) > 1):
            ignored_rhos = 0.0
            for remaining_length in range(len(self), 0, -1):
                upper_cutoff = self[remaining_length - 1].rho
                cut_rho = upper_cutoff * remaining_length + ignored_rhos
                if cut_rho > (1.0 - remaining_rho):
                    density_cut = (1.0 - remaining_rho - ignored_rhos) / remaining_length
                    break
                else:
                    ignored_rhos += upper_cutoff
            del self[remaining_length:]
            for el_id in range(remaining_length):
                self[el_id].rho = max(
                    0.0, self[el_id].rho - density_cut
                )  # max was introduced in case there is some weird numerical noise.
            self.normalize_rhos()

    def sort_cutoff_minor_weights(self, remaining_rho=None):
        self.sort_rhos()
        self.cutoff_minor_weights(remaining_rho=remaining_rho)


# Misc procedures for list and dictionnary handling.
def any_element_in_list(list_in, *els):
    for el in els:
        if el in list_in:
            return True
    return False


def repeated_dict(labels, repeated_el, copy_needed=False):
    output = {}
    for l in labels:
        if copy_needed:
            output[l] = copy.deepcopy(repeated_el)
        else:
            output[l] = repeated_el
    return output


def all_None_dict(labels):
    return repeated_dict(labels, None)


def lookup_or_none(dict_in, key):
    if key in dict_in:
        return dict_in[key]
    else:
        return None


# We evaluate a lot of logarithms of natural numbers, so makes sense to cut down on their calculation.
def intlog_no_precalc(int_in):
    return np.log(float(int_in))


class NaturalLogLookup:
    def __init__(self):
        self.saved_values = None
        self.max_avail_val = 0

    def gen_new_saved_values(self, new_max_avail_val):
        new_saved_values = np.empty((new_max_avail_val,))
        if self.saved_values is not None:
            new_saved_values[: self.max_avail_val] = self.saved_values[:]
        old_max_avail_val = self.max_avail_val
        self.max_avail_val
        return new_saved_values, old_max_avail_val

    def fill_saved_values(self, new_max_avail_val):
        new_saved_values, old_max_avail_val = self.gen_new_saved_values(new_max_avail_val)
        for i in range(old_max_avail_val, new_max_avail_val):
            new_saved_values[i] = intlog_no_precalc(i + 1)
        self.saved_values = new_saved_values
        self.max_avail_val = new_max_avail_val

    def __call__(self, int_val):
        assert int_val > 0
        if int_val > self.max_avail_val:
            self.fill_saved_values(int_val)
        return self.saved_values[int_val - 1]


intlog = NaturalLogLookup()


def llenlog(l):
    return intlog(len(l))


class FactorialLogLookup(NaturalLogLookup):
    def fill_saved_values(self, new_max_avail_val):
        new_saved_values, old_max_avail_val = self.gen_new_saved_values(new_max_avail_val)
        if old_max_avail_val == 1:
            current_log = 0.0
        else:
            current_log = self.saved_values[old_max_avail_val - 1]
        for i in range(old_max_avail_val, new_max_avail_val):
            current_log += intlog(i + 1)
            new_saved_values[i] = current_log
        self.saved_values = new_saved_values
        self.max_avail_val = new_max_avail_val


log_natural_factorial = NaturalLogLookup()


# Used for different random choices (mainly in modify)
def available_options_prob_norm(dict_in):
    output = 0.0
    for i in list(dict_in):
        if len(dict_in[i]) != 0:
            output += 1.0
    return output


def random_choice_from_dict(possibilities, choices=None, get_probability_of=None):
    prob_sum = 0.0
    corr_prob_choice = {}
    if choices is None:
        choices = possibilities.keys()
    for choice in choices:
        if (choice not in possibilities) or (len(possibilities[choice]) == 0):
            continue
        if isinstance(choices, dict):
            prob = choices[choice]
        else:
            prob = 1.0
        prob_sum += prob
        corr_prob_choice[choice] = prob
    if get_probability_of is None:
        if len(corr_prob_choice.keys()) == 0:
            raise Exception("Something is wrong: encountered a molecule that cannot be changed")
        final_choice = random.choices(
            list(corr_prob_choice.keys()), list(corr_prob_choice.values())
        )[0]
        final_prob_log = np.log(corr_prob_choice[final_choice] / prob_sum)
        return final_choice, possibilities[final_choice], final_prob_log
    else:
        return possibilities[get_probability_of], np.log(
            corr_prob_choice[get_probability_of] / prob_sum
        )


def random_choice_from_nested_dict(possibilities, choices=None, get_probability_of=None):
    continue_nested = True
    cur_possibilities = possibilities
    prob_log = 0.0
    cur_choice_prob_dict = choices

    if get_probability_of is None:
        final_choice = []
    else:
        choice_level = 0

    while continue_nested:
        if not isinstance(cur_possibilities, dict):
            if isinstance(cur_possibilities, list):
                prob_log -= llenlog(cur_possibilities)
                if get_probability_of is None:
                    final_choice.append(random.choice(cur_possibilities))
            break
        if get_probability_of is None:
            get_prob_arg = None
        else:
            get_prob_arg = get_probability_of[choice_level]
            choice_level += 1
        rcfd_res = random_choice_from_dict(
            cur_possibilities,
            choices=cur_choice_prob_dict,
            get_probability_of=get_prob_arg,
        )
        prob_log += rcfd_res[-1]
        cur_possibilities = rcfd_res[-2]
        cur_choice_prob_dict = None
        if get_probability_of is None:
            final_choice.append(rcfd_res[0])
    if get_probability_of is None:
        return final_choice, prob_log
    else:
        return prob_log


def list2colors(obj_list):
    """
    Color obj_list in a way that each equal obj1 and obj2 were the same color.
    Used for defining canonical permutation of a graph.
    """
    ids_objs = list(enumerate(obj_list))
    ids_objs.sort(key=lambda x: x[1])
    num_obj = len(ids_objs)
    colors = np.zeros(num_obj, dtype=int)
    cur_color = 0
    prev_obj = ids_objs[0][1]
    for i in range(1, num_obj):
        cur_obj = ids_objs[i][1]
        if cur_obj != prev_obj:
            cur_color += 1
            prev_obj = cur_obj
        colors[ids_objs[i][0]] = cur_color
    return colors


def merge_unrepeated_sorted_lists(base_list, added_list):
    for el in added_list:
        if el not in base_list:
            base_list.add(el)


# from: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-equal
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
