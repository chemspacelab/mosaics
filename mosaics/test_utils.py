# Utils that mainly appear in tests or analysis of data.
import copy
import datetime
import os

import numpy as np
from joblib import Parallel, delayed
from sortedcontainers import SortedDict

from .data import NUCLEAR_CHARGE
from .modify import egc_change_func, random_modification_path_choice
from .random_walk import (
    CandidateCompound,
    RandomWalk,
    TrajectoryPoint,
    default_minfunc_name,
    egc_valid_wrt_change_params,
    full_change_list,
    inverse_procedure,
    randomized_change,
)
from .utils import run
from .valence_treatment import ChemGraph, str2ChemGraph


def elements_in_egc_list(egc_list, as_elements=True):
    nuclear_charges = []
    for egc in egc_list:
        for nc in egc.true_ncharges():
            if nc not in nuclear_charges:
                nuclear_charges.append(nc)
    nuclear_charges.sort()
    if as_elements:
        output = []
        for el, nc in NUCLEAR_CHARGE.items():
            if nc in nuclear_charges:
                output.append(el)
    else:
        output = nuclear_charges
    return output


def egc_list_sizes(egc_list):
    sizes = []
    for egc in egc_list:
        sizes.append((egc.num_heavy_atoms(), egc.num_atoms()))
    return sizes


def egc_list_nhatoms_hist(egc_list):
    histogram = {}
    for egc in egc_list:
        nhatoms = egc.num_heavy_atoms()
        if nhatoms not in histogram:
            histogram[nhatoms] = 0
        histogram[nhatoms] += 1
    return SortedDict(histogram)


# For numerically verifying that detailed balance is satisfied for individual moves.
def crossover_move_attempt(tp_init, **randomized_change_params):
    rw = RandomWalk(init_egcs=[tp.egc for tp in tp_init], **randomized_change_params)
    pair, prob_balance = rw.trial_crossover_MC_step([0, 1])

    if pair is None:
        tpair = None
    else:
        tpair = tuple(pair)

    return tpair, prob_balance


str_randomized_change_params = "randomized_change_params"


def tp_not_valid(tp, **randomized_change_params):
    if str_randomized_change_params in randomized_change_params:
        return not egc_valid_wrt_change_params(
            tp.egc, **randomized_change_params[str_randomized_change_params]
        )
    else:
        return False


def tps_not_valid(tps, **randomized_change_params):
    if isinstance(tps, TrajectoryPoint):
        return tp_not_valid(tps, **randomized_change_params)
    else:
        for tp in tps:
            if tp_not_valid(tp, **randomized_change_params):
                return True
        return False


trial_attempt_funcs = {
    TrajectoryPoint: randomized_change,
    tuple: crossover_move_attempt,
}


def calc_bin_id(x, bin_size=None):
    if bin_size is None:
        return 0
    if np.abs(x) < 0.5 * bin_size:
        return 0
    output = int(np.abs(x) / bin_size - 0.5)
    if x < 0.0:
        output *= -1
    return output


def serial_generate_attempts(tp_init, attempt_func, num_attempts, **randomized_change_params):
    return [attempt_func(tp_init, **randomized_change_params) for _ in range(num_attempts)]


def generate_attempts(
    tp_init, attempt_func, num_attempts, nprocs=None, **randomized_change_params
):
    if nprocs is None:
        return serial_generate_attempts(
            tp_init, attempt_func, num_attempts, **randomized_change_params
        )
    job_num_attempts = num_attempts // nprocs
    all_num_attempts = [job_num_attempts for _ in range(nprocs)]
    all_num_attempts[0] += num_attempts % nprocs
    output = Parallel(n_jobs=nprocs)(
        delayed(serial_generate_attempts)(
            tp_init, attempt_func, cur_num_attempts, **randomized_change_params
        )
        for cur_num_attempts in all_num_attempts
    )
    straightened_output = []
    for l in output:
        straightened_output += l
    return straightened_output


def check_one_sided_prop_probability(
    tp_init, tp_trial, num_attempts=10000, bin_size=None, nprocs=None, **randomized_change_params
):
    if isinstance(tp_trial, list):
        true_list = tp_trial
    else:
        true_list = [tp_trial]
    attempt_func = trial_attempt_funcs[type(tp_init)]

    est_balances = [[] for _ in true_list]
    tp_new_prob_balance_vals = generate_attempts(
        tp_init, attempt_func, num_attempts, nprocs=nprocs, **randomized_change_params
    )
    for tp_new, prob_balance in tp_new_prob_balance_vals:
        if tp_new is None:
            continue
        if tp_new not in true_list:
            continue

        if tps_not_valid(tp_new, **randomized_change_params):
            raise Exception

        i = true_list.index(tp_new)
        est_balances[i].append(prob_balance)

    output = []
    for est_bal in est_balances:
        trial_prob_arrays = {}
        observed_probs = {}
        for bal in est_bal:
            bin_id = calc_bin_id(bal, bin_size)
            if bin_id not in trial_prob_arrays:
                trial_prob_arrays[bin_id] = []
                observed_probs[bin_id] = 0.0
            trial_prob_arrays[bin_id].append(bal)
            observed_probs[bin_id] += 1.0
        trial_prob_averages = {}
        for bin_id, prob_arr in trial_prob_arrays.items():
            trial_prob_averages[bin_id] = (np.mean(prob_arr), np.std(prob_arr))
            observed_probs[bin_id] /= num_attempts
        output.append((trial_prob_averages, observed_probs))

    if isinstance(tp_trial, list):
        return output
    else:
        return output[0]


def log_ratio_rmse(num_attempts, *probs):
    sqdev = sum(1.0 / prob - 1.0 for prob in probs)
    return np.sqrt(sqdev / num_attempts)


def check_prop_probability(tp1, tp2_list, label_dict=None, num_attempts=10000, **one_sided_kwargs):
    """
    Check that simple MC moves satisfy detailed balance for a pair of trajectory point objects.
    """
    if isinstance(tp2_list, list):
        true_list = tp2_list
    else:
        true_list = [tp2_list]
    print("INITIAL MOLECULE:", tp1)
    forward_results = check_one_sided_prop_probability(
        tp1, true_list, num_attempts=num_attempts, **one_sided_kwargs
    )
    for tp2, (forward_trial_prob_averaged, forward_observed_probs) in zip(
        true_list, forward_results
    ):
        print("CASE:", tp2)
        if label_dict is not None:
            print("CASE LABEL:", label_dict[str(tp2)])
        (
            inverse_trial_prob_averaged,
            inverse_observed_probs,
        ) = check_one_sided_prop_probability(
            tp2, tp1, num_attempts=num_attempts, **one_sided_kwargs
        )
        hist_ids = list(inverse_observed_probs.keys())
        for forward_hist_id in forward_observed_probs.keys():
            inverted_fhi = -forward_hist_id
            if inverted_fhi not in hist_ids:
                hist_ids.append(inverted_fhi)
        for hist_id in hist_ids:
            print("BIN ID:", hist_id)
            inverted_fhi = -hist_id
            forward_present = inverted_fhi in forward_trial_prob_averaged
            inverse_present = hist_id in inverse_trial_prob_averaged
            if inverse_present:
                inverse_prob = inverse_observed_probs[hist_id]
                print("INVERSE:", *inverse_trial_prob_averaged[hist_id], inverse_prob)
            else:
                print("NO INVERSE STEPS")
            if forward_present:
                forward_prob = forward_observed_probs[inverted_fhi]
                print("FORWARD:", *forward_trial_prob_averaged[inverted_fhi], forward_prob)
            else:
                print("NO FORWARD STEPS")
            if forward_present and inverse_present:
                print(
                    "OBSERVED RATIO:",
                    np.log(forward_prob / inverse_prob),
                    "pm",
                    log_ratio_rmse(num_attempts, forward_prob, inverse_prob),
                )


def generate_proc_example(
    tp: TrajectoryPoint,
    change_procedure,
    new_tp=None,
    print_dicts=False,
    max_attempt_num=10000,
    **other_kwargs
):
    tp_copy = copy.deepcopy(tp)
    tp_copy.possibility_dict = None
    tp_copy.init_possibility_info(change_prob_dict=[change_procedure], **other_kwargs)
    if change_procedure not in tp_copy.possibility_dict:
        return None
    tp_copy.modified_possibility_dict = copy.deepcopy(tp_copy.possibility_dict)
    for _ in range(max_attempt_num):
        old_egc = tp.egc
        possibilities = tp_copy.modified_possibility_dict[change_procedure]
        modification_path, _ = random_modification_path_choice(
            old_egc, possibilities, change_procedure, **other_kwargs
        )
        new_egc = egc_change_func(tp_copy.egc, modification_path, change_procedure, **other_kwargs)
        if new_egc is not None:
            tp_out = TrajectoryPoint(egc=new_egc)
            if new_tp is not None:
                if tp_out != new_tp:
                    continue
            if print_dicts:
                inv_proc = inverse_procedure[change_procedure]
                tp_out.init_possibility_info(change_prob_dict=[inv_proc], **other_kwargs)
                print("EXAMPLE FOR:", tp_copy, change_procedure)
                print("NEW TP:", tp_out)
                print("INVERSE PROC DICT:", tp_out.possibility_dict[inv_proc])
                print("FORWARD PROC DICT:", tp_copy.possibility_dict[change_procedure])
                tp_out.possibility_dict = None
            return tp_out
    return None


def generate_proc_sample_dict(tp_init, change_prob_dict=full_change_list, **other_kwargs):
    l = []
    d = {}
    for change_procedure in change_prob_dict:
        tp_new = generate_proc_example(tp_init, change_procedure, **other_kwargs)
        if tp_new is not None:
            l.append(tp_new)
            d[str(tp_new)] = change_procedure
    return l, d


def all_procedure_prop_probability_checks(tp_init, num_attempts=10000, **other_kwargs):
    l, d = generate_proc_sample_dict(tp_init, **other_kwargs)
    check_prop_probability(tp_init, l, label_dict=d, num_attempts=num_attempts, **other_kwargs)


# For checking that the distribution follows Boltzmann distribution.
def distr_bin_index(val, val_lbound=None, val_ubound=None, num_bins=1):
    if val_lbound is None:
        return 0
    else:
        return int(num_bins * (val - val_lbound) / (val_ubound - val_lbound))


def print_distribution_analysis(
    histogram,
    betas,
    min_func_name=default_minfunc_name,
    val_lbound=None,
    val_ubound=None,
    num_bins=1,
):
    """
    Analyse the distribution in the histogram and compare it to Boltzmann weights.
    """
    nmols = np.zeros((num_bins,), dtype=int)
    tot_min = None
    tot_max = None
    for tp in histogram:
        cur_f = tp.calculated_data[min_func_name]
        if cur_f is None:
            continue
        if (tot_min is None) or (cur_f < tot_min):
            tot_min = cur_f
        if (tot_max is None) or (cur_f > tot_max):
            tot_max = cur_f
        cur_bin_id = distr_bin_index(
            cur_f, val_lbound=val_lbound, val_ubound=val_ubound, num_bins=num_bins
        )
        nmols[cur_bin_id] += 1
    print("Range of minimized function values:", tot_min, tot_max)
    print("TOTAL NUMBERS OF CONSIDERED MOLECULES:")
    for bin_id, nmol in enumerate(nmols):
        print(bin_id, nmol)
    for beta_id, beta in enumerate(betas):
        print("DATA FOR REPLICA", beta_id, "BETA:", beta)
        print_distribution_analysis_single_beta(
            histogram,
            beta_id,
            beta,
            min_func_name,
            val_lbound=val_lbound,
            val_ubound=val_ubound,
            num_bins=num_bins,
        )


def print_distribution_analysis_single_beta(
    histogram,
    beta_id,
    beta,
    min_func_name,
    val_lbound=None,
    val_ubound=None,
    num_bins=1,
):
    """
    Analyse the distribution in the histogram and compare it to Boltzmann weights for one beta.
    histogram : histogram to be analyzed, list of TrajectoryPoint objects
    beta_id : index of the replica to be analyzed
    beta : inverse beta
    min_func_name : name of the minimized function
    num_bins : number of bins into which values are sorted
    val_lbound : upper bound of values to be considered
    val_ubound : lower bound of values to be considered
    """
    if (val_lbound is None) or (val_ubound is None):
        num_bins = 1
    distr = np.zeros((num_bins,))
    distr2 = np.zeros((num_bins,))
    present = np.zeros((num_bins,), dtype=bool)
    nmols = np.zeros((num_bins,), dtype=int)
    bin_middles = np.zeros((num_bins,))
    if val_lbound is None:
        bin_middles[0] = 0.0
    else:
        for i in range(num_bins):
            bin_middles[i] = val_lbound + (i + 0.5) * (val_ubound - val_lbound) / num_bins
    #    bin_middle_boltzmann_weights=np.exp(-beta*bin_middles)
    for tp in histogram:
        cur_f = tp.calculated_data[min_func_name]
        if cur_f is None:
            continue
        cur_bin_id = distr_bin_index(
            cur_f, val_lbound=val_lbound, val_ubound=val_ubound, num_bins=num_bins
        )
        cur_d = tp.visit_num(beta_id)
        if cur_d > 0.5:
            present[cur_bin_id] = True
        cur_r = cur_d
        if beta is not None:
            cur_r *= np.exp(beta * (cur_f - bin_middles[cur_bin_id]))
        distr[cur_bin_id] += cur_r
        distr2[cur_bin_id] += cur_r**2
        nmols[cur_bin_id] += 1

    stddevs = np.zeros((num_bins,))
    errs = np.zeros((num_bins,))
    rel_errs = np.zeros((num_bins,))
    min_err_id = None
    min_err = None

    for i in np.where(present)[0]:
        distr[i] /= nmols[i]
        distr2[i] /= nmols[i]
        stddevs[i] = np.sqrt(distr2[i] - distr[i] ** 2)
        errs[i] = stddevs[i] / np.sqrt(nmols[i])
        rel_errs[i] = errs[i] / distr[i]
        if (min_err is None) or (min_err > rel_errs[i]) and (nmols[i] != 1):
            min_err = rel_errs[i]
            min_err_id = i

    if min_err_id is None:
        min_err_id = np.argmax(distr)

    print(
        "Distributions: (bin index : bin middle : adjusted probability density : standard deviation : number of molecules found)"
    )
    for bin_id, (bin_middle, d, stddev, nmol) in enumerate(
        zip(bin_middles, distr, stddevs, nmols)
    ):
        print(bin_id, bin_middle, d, stddev, nmol)
    if beta is not None:
        print(
            "Deviation of probability densities from Boltzmann factors: (bin index : deviation : estimated error)"
        )
        for i in np.where(present)[0]:
            dev = np.log(distr[i] / distr[min_err_id]) + beta * (
                bin_middles[i] - bin_middles[min_err_id]
            )
            if i == min_err_id:
                dev_err = 0.0
            else:
                dev_err = np.sqrt(rel_errs[i] ** 2 + rel_errs[min_err_id] ** 2)
            print(i, dev, dev_err)


def ordered_filename(file_id, print_file_prefix, print_file_suffix=".txt"):
    return print_file_prefix + str(file_id) + print_file_suffix


def print_to_separate_file_wprefix(printed_string, print_file_prefix, print_file_suffix=".txt"):
    """
    Write string into the next file with a name derived as print_file_prefix+{file_id}+print_file_suffix
    """
    file_id = 0
    filename = ordered_filename(file_id, print_file_prefix, print_file_suffix=print_file_suffix)
    while os.path.isfile(
        ordered_filename(file_id, print_file_prefix, print_file_suffix=print_file_suffix)
    ):
        file_id += 1
        filename = ordered_filename(
            file_id, print_file_prefix, print_file_suffix=print_file_suffix
        )
    with open(filename, "w") as f:
        f.write(printed_string)


# Relative float difference that should be ignored when comparing tests.
negligible_float_relative_difference = 1e-6


def set_negligible_float_relative_difference(new_negligible_float_relative_difference):
    global negligible_float_relative_difference
    negligible_float_relative_difference = new_negligible_float_relative_difference


# How floats are written. Important to ensure enough data is saved to check whether difference is negligible.
test_float_str_format = "{:.8e}"


def set_test_float_str_format(new_test_float_str_format):
    global test_float_str_format
    test_float_str_format = new_test_float_str_format


# Labels for some data classes that can be written by SimulationLogIO.
timestamp_label = "TIME"
int_label = "INT"
float_label = "FLT"
bool_label = "BOOL"
chemgraph_label = "CHEMGRAPH"
trajectory_point_label = "TRAJECTORYPOINT"
candidate_compound_label = "CANDIDATECOMPOUND"


data_str_labels = {
    datetime.datetime: timestamp_label,
    ChemGraph: chemgraph_label,
    TrajectoryPoint: trajectory_point_label,
    CandidateCompound: candidate_compound_label,
    int: int_label,
    float: float_label,
    bool: bool_label,
    np.float64: float_label,
    np.int64: int_label,
    np.bool_: bool_label,
}


def tp2tuple(tp: TrajectoryPoint) -> tuple:
    use_first_encounter = tp.first_global_MC_step_encounter
    if use_first_encounter is None:
        use_first_encounter = -1
    return (int(use_first_encounter), tp.chemgraph())


def cc2tuple(cc: CandidateCompound) -> tuple:
    return (float(cc.func_val), *tp2tuple(cc.tp))


data_comp_obj_creator = {
    np.float64: float,
    np.int64: int,
    np.bool_: bool,
    TrajectoryPoint: tp2tuple,
    CandidateCompound: cc2tuple,
}

tuple_str_delimiter = "!"


def str2TrajectoryPoint_comp_obj(str_in=None, split_str=None):
    if split_str is None:
        split_str = str_in.split(tuple_str_delimiter)
    return (int(split_str[0]), str2ChemGraph(split_str[1]))


def str2CandidateCompound_comp_obj(str_in):
    split_str = str_in.split(tuple_str_delimiter)
    func_val = float(split_str[0])
    return (func_val, *str2TrajectoryPoint_comp_obj(split_str=split_str[1:]))


def obj2comparison_obj(obj):
    """
    For some objects it is more convenient to convert them to something else.
    """
    cur_type = type(obj)
    if cur_type in data_comp_obj_creator:
        return data_comp_obj_creator[cur_type](obj)
    else:
        return obj


def print_test(obj):
    comp_obj = obj2comparison_obj(obj)
    if isinstance(comp_obj, tuple):
        return tuple_str_delimiter.join(print_test(el) for el in comp_obj)
    if isinstance(comp_obj, float):
        return test_float_str_format.format(comp_obj)
    return str(comp_obj)


def str2bool(str_in):
    return str(True) == str_in


data_str_label_conv_functions = {
    chemgraph_label: str2ChemGraph,
    trajectory_point_label: str2TrajectoryPoint_comp_obj,
    candidate_compound_label: str2CandidateCompound_comp_obj,
    int_label: int,
    float_label: float,
    bool_label: str2bool,
}


def compared_objects_identical(obj1, obj2):
    if type(obj1) is not type(obj2):
        return False
    if isinstance(obj1, float):
        try:
            rel_diff = 2 * (obj1 - obj2) / (abs(obj1) + abs(obj2))
        except ZeroDivisionError:
            rel_diff = 0.0
        return rel_diff < negligible_float_relative_difference
    if isinstance(obj1, tuple):
        if len(obj1) != len(obj2):
            return False
        for el1, el2 in zip(obj1, obj2):
            if not compared_objects_identical(el1, el2):
                return False
        return True
    return obj1 == obj2


def now():
    return datetime.datetime.now()


# NOTE: The SimulationLogIO class was designed with saving tests output in plain text (rather than binary) form
# because I like to keep things human-readable and independent from, for example, changes in definitions of ChemGraph or TrajectoryPoint classes.
# I'm not %100 sure it's the best course of action.


class SimulationLogIO:
    def __init__(
        self,
        save_printed=True,
        filename=None,
        benchmark_filename=None,
        exception_on_failure=True,
    ):
        """
        Auxiliary class used to read and write test files in a way invariant to the canonical ordering currently used.
        """
        self.entry_list = []
        self.save_printed = save_printed
        self.filename = filename
        if filename is not None:
            run("rm", "-f", filename)
        self.io = None
        self.difference_encountered = False
        self.exception_on_failure = exception_on_failure

        self.benchmark_filename = benchmark_filename
        if (self.benchmark_filename is not None) and (os.path.isfile(self.benchmark_filename)):
            self.benchmark_entry_list = self.import_from(self.benchmark_filename)
            print("(BENCHMARK IMPORTED)")
            self.current_list_id = -1
        else:
            self.benchmark_entry_list = None
            self.current_entry_id = None

    def print_timestamp(self, comment=None):
        self.print(now(), comment=comment)

    def difference_was_encountered(self):
        self.difference_encountered = True
        if self.exception_on_failure:
            raise Exception

    def check_benchmark_agreement(self, new_list, sorted_comparison=False):
        if self.benchmark_entry_list is None:
            return
        self.current_list_id += 1
        if self.current_list_id > len(self.benchmark_entry_list):
            print("DIFF: BENCHMARK_ENDED")
            self.difference_encountered = True
        cur_benchmark = self.benchmark_entry_list[self.current_list_id]

        checked_new_list = [obj2comparison_obj(el) for el in new_list]

        if sorted_comparison:
            checked_new_list[2:] = sorted(checked_new_list[2:])
            cur_benchmark[2:] = sorted(cur_benchmark[2:])

        if len(checked_new_list) != len(cur_benchmark):
            print("DIFF: DIFFERENT_LENGTHS")
            self.difference_was_encountered()
        for new_el, bench_el in zip(checked_new_list, cur_benchmark):
            if not compared_objects_identical(new_el, bench_el):
                print("DIFF:", new_el, bench_el)
                self.difference_was_encountered()
            if isinstance(new_el, str) and (new_el == timestamp_label):
                break

    def print_list(self, printed_data, comment=None, sorted_comparison=False):
        if comment is None:
            new_list = ["###"]
        else:
            new_list = [comment]
        data_str_label = data_str_labels[type(printed_data[0])]
        if self.save_printed:
            new_list.append(data_str_label)

        new_list += list(printed_data)

        self.entry_list.append(new_list)
        self.check_benchmark_agreement(new_list, sorted_comparison=sorted_comparison)

        printed_ios = [None]
        if self.filename is not None:
            self.io = open(self.filename, "a")
            printed_ios.append(self.io)
        printed_list = [print_test(el) for el in new_list]
        for printed_io in printed_ios:
            print(*printed_list, file=printed_io)
        if self.filename is not None:
            self.io.close()

    def print(self, *data, comment=None, sorted_comparison=False):
        self.print_list(data, comment=comment, sorted_comparison=sorted_comparison)

    def import_from(self, filename):
        combined_lists = []
        input_file = open(filename, "r")
        for line in input_file.readlines():
            spl_line = line.strip().split()
            label = spl_line[1]
            new_list = spl_line[:2]
            # Timestamps are irrelevant for benchmark comparison.
            if label == timestamp_label:
                new_list.append("DUMMY")
            else:
                str_converter = data_str_label_conv_functions[label]
                for data_line in spl_line[2:]:
                    new_list.append(str_converter(data_line))
            combined_lists.append(new_list)
        input_file.close()
        return combined_lists

    def __eq__(self, other_sli):
        return self.entry_list == other_sli.entry_list
