# Procedures used for assigning beta parameters.
import numpy as np


def gen_beta_array(num_greedy_replicas, max_real_beta, *args):
    """
    Generate piecewise-linear beta array. Arguments should be written in order:
    number of greedy replicas , (linear inverval start , linear interval end) * (number of linear intervals)
    Each linear interval finishes at the start of the next one; the last interval finishes at zero.
    """
    output = [None for _ in range(num_greedy_replicas)]
    if len(args) == 0:
        num_intervals = 1
    else:
        if len(args) % 2 == 0:
            raise Exception("Wrong gen_beta_array arguments")
        num_intervals = (len(args) + 1) // 2
    for interval_id in range(num_intervals):
        if interval_id == 0:
            cur_max_rb = max_real_beta
        else:
            cur_max_rb = args[interval_id * 2 - 1]
        if interval_id == num_intervals - 1:
            cur_min_rb = 0.0
        else:
            cur_min_rb = args[interval_id * 2 + 1]
        if len(args) == 0:
            num_frac_betas = 0
        else:
            cur_beta_delta = args[interval_id * 2]
            num_frac_betas = max(
                int(
                    (float(cur_max_rb) - float(cur_min_rb)) / float(cur_beta_delta)
                    - 0.01
                ),
                0,
            )
        added_beta = float(cur_max_rb)
        output.append(added_beta)
        for _ in range(num_frac_betas):
            added_beta -= cur_beta_delta
            output.append(added_beta)
    return output


def gen_inv_lin_beta_array(
    num_greedy_replicas: int, min_real_beta: float, num_real_betas: int
):
    """
    Generate an array of betas whose inverse changes linearly from 0 (corresponding to the greedy replicas) to a certain value.
    num_greedy_replicas : number of "greedy" replicase
    min_real_beta : minimal value of beta (highest temperature)
    num_real_betas : how many real beta values we want in the array
    """
    output = [None for _ in range(num_greedy_replicas)]
    max_real_beta = min_real_beta * num_real_betas
    for real_beta_id in range(num_real_betas):
        output.append(max_real_beta / (real_beta_id + 1))
    return output


def gen_exp_lin_beta_array(
    num_greedy_replicas: int,
    min_real_beta: float,
    num_real_betas: int,
    end_beta_ratio: float,
):
    """
    Generate an array of betas following the formula:
    beta_i=C1 / (C2 ** i - 1)     (i=1, ..., num_real_betas)
    C1 and C2 are chosen in such a way that beta_{num_real_betas}=min_real_beta and beta_{num_real_betas-1}/beta_{num_real_betas}=end_beta_ratio
    """
    assert num_real_betas > 1
    # First determine C_1 and C_2
    C_polynom_coeffs = np.ones((num_real_betas,))
    C_polynom_coeffs[1:] -= end_beta_ratio
    r = np.roots(C_polynom_coeffs)
    C2 = np.real(sorted(r, key=lambda x: np.abs(np.imag(x)))[0])
    assert C2 > 0.0  # just in case
    C1 = min_real_beta * (C2**num_real_betas - 1)
    output = [None for _ in range(num_greedy_replicas)]
    for i in range(num_real_betas):
        output.append(C1 / (C2 ** (i + 1) - 1))
    return output


def gen_exp_beta_array(
    num_greedy_replicas: int,
    min_real_beta: float,
    num_real_betas: int,
    max_real_beta: float or None = None,
    real_beta_multiplier: float or None = None,
):
    """
    Generate a range of beta values where the real beta values are separated evenly in logspace.
    """
    output = [None for _ in range(num_greedy_replicas)]
    if num_real_betas == 1:
        output.append(min_real_beta)
        return output
    log_min_real_beta = np.log2(min_real_beta)
    if max_real_beta is None:
        if num_real_betas != 1:
            assert real_beta_multiplier is not None
            log_max_real_beta = log_min_real_beta + (num_real_betas - 1) * np.log2(
                real_beta_multiplier
            )
    else:
        log_max_real_beta = np.log2(max_real_beta)
    if num_real_betas == 1:
        real_betas = [min_real_beta]
    else:
        real_betas = np.logspace(
            log_max_real_beta, log_min_real_beta, num=num_real_betas, base=2
        )

    output += list(real_betas)
    return output
