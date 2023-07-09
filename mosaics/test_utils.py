# Utils that mainly appear in tests or analysis of data.
from ..data import NUCLEAR_CHARGE
from sortedcontainers import SortedDict
from .random_walk import (
    randomized_change,
    TrajectoryPoint,
    RandomWalk,
    full_change_list,
    random_choice_from_nested_dict,
    egc_change_func,
    inverse_procedure,
    default_minfunc_name,
)
import numpy as np
import copy


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
def genetic_move_attempt(tp_init, **randomized_change_params):
    rw = RandomWalk(init_egcs=[tp.egc for tp in tp_init], **randomized_change_params)
    pair, prob_balance = rw.trial_genetic_MC_step([0, 1])
    if pair is None:
        tpair = None
    else:
        tpair = tuple(pair)
    return tpair, prob_balance


trial_attempt_funcs = {TrajectoryPoint: randomized_change, tuple: genetic_move_attempt}


def calc_bin_id(x, bin_size=None):
    if bin_size is None:
        return 0
    if np.abs(x) < 0.5 * bin_size:
        return 0
    output = int(np.abs(x) / bin_size - 0.5)
    if x < 0.0:
        output *= -1
    return output


def check_one_sided_prop_probability(
    tp_init, tp_trial, num_attempts=10000, bin_size=None, **randomized_change_params
):
    if isinstance(tp_trial, list):
        true_list = tp_trial
    else:
        true_list = [tp_trial]
    attempt_func = trial_attempt_funcs[type(tp_init)]

    est_balances = [[] for _ in true_list]
    for _ in range(num_attempts):
        tp_new, prob_balance = attempt_func(tp_init, **randomized_change_params)
        if tp_new is None:
            continue
        if tp_new not in true_list:
            continue
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


def check_prop_probability(
    tp1, tp2_list, label_dict=None, num_attempts=10000, **one_sided_kwargs
):
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
                print(
                    "FORWARD:", *forward_trial_prob_averaged[inverted_fhi], forward_prob
                )
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
    tp, change_procedure, new_tp=None, print_dicts=False, **other_kwargs
):
    tp_copy = copy.deepcopy(tp)
    tp_copy.possibility_dict = None
    tp_copy.init_possibility_info(change_prob_dict=[change_procedure], **other_kwargs)
    tp_copy.modified_possibility_dict = copy.deepcopy(tp_copy.possibility_dict)
    while tp_copy.modified_possibility_dict:
        modification_path, _ = random_choice_from_nested_dict(
            tp_copy.modified_possibility_dict[change_procedure]
        )
        new_egc = egc_change_func(
            tp_copy.egc, modification_path, change_procedure, **other_kwargs
        )
        if new_egc is not None:
            tp_out = TrajectoryPoint(egc=new_egc)
            if new_tp is not None:
                if tp_out != new_tp:
                    continue
            if print_dicts:
                inv_proc = inverse_procedure[change_procedure]
                tp_out.init_possibility_info(
                    change_prob_dict=[inv_proc], **other_kwargs
                )
                print("EXAMPLE FOR:", tp_copy, change_procedure)
                print("NEW TP:", tp_out)
                print("INVERSE PROC DICT:", tp_out.possibility_dict[inv_proc])
                print("FORWARD PROC DICT:", tp_copy.possibility_dict[change_procedure])
                tp_out.possibility_dict = None
            return tp_out
        tp_copy.delete_mod_path([change_procedure, *modification_path])
    return None


def generate_proc_sample_dict(
    tp_init, change_prob_dict=full_change_list, **other_kwargs
):
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
    check_prop_probability(
        tp_init, l, label_dict=d, num_attempts=num_attempts, **other_kwargs
    )


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
            bin_middles[i] = (
                val_lbound + (i + 0.5) * (val_ubound - val_lbound) / num_bins
            )
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
