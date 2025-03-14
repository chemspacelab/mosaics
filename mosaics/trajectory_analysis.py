import numpy as np

from .modify import global_step_traj_storage_label


class DataUnavailable(Exception):
    """
    Raised if data not available in a histogram is referred to.
    """


# Some procedures for convenient RandomWalk analysis.
def histogram_num_replicas(histogram):
    max_replica_id = 0
    for tp in histogram:
        cur_visit_step_ids = tp.visit_step_ids
        if global_step_traj_storage_label not in cur_visit_step_ids:
            continue
        visiting_replicas = cur_visit_step_ids[global_step_traj_storage_label].keys()
        if visiting_replicas:
            max_replica_id = max(max_replica_id, max(visiting_replicas))
    return max_replica_id + 1


def ordered_trajectory_ids(histogram, global_MC_step_counter=None, num_replicas=None):
    if num_replicas is None:
        num_replicas = histogram_num_replicas(histogram)
        if num_replicas is None:
            raise DataUnavailable
    if global_MC_step_counter is None:
        global_MC_step_counter = 0
        for tp in histogram:
            if global_step_traj_storage_label in tp.visit_step_ids:
                for visit_ids, num_visits in zip(
                    tp.visit_step_ids[global_step_traj_storage_label].values(),
                    tp.visit_step_num_ids[global_step_traj_storage_label].values(),
                ):
                    if num_visits > 0:
                        global_MC_step_counter = max(
                            global_MC_step_counter, visit_ids[num_visits - 1]
                        )
    output = np.zeros((global_MC_step_counter + 1, num_replicas), dtype=int)
    output[:, :] = -1
    for tp_id, tp in enumerate(histogram):
        if global_step_traj_storage_label in tp.visit_step_ids:
            cur_vsi_dict = tp.visit_step_ids[global_step_traj_storage_label]
            cur_vsni_dict = tp.visit_step_num_ids[global_step_traj_storage_label]
            for replica_id, num_visits in cur_vsni_dict.items():
                visits = cur_vsi_dict[replica_id][:num_visits]
                for v in visits:
                    output[v, replica_id] = tp_id
    for step_id in range(global_MC_step_counter):
        true_step_id = step_id + 1
        for replica_id in range(num_replicas):
            if output[true_step_id, replica_id] == -1:
                output[true_step_id, replica_id] = output[true_step_id - 1, replica_id]
    return output


def ordered_trajectory(histogram, **ordered_trajectory_ids_kwargs):
    output = []
    for tp_ids in ordered_trajectory_ids(histogram, **ordered_trajectory_ids_kwargs):
        output.append([histogram[tp_id] for tp_id in tp_ids])
    return output


def ordered_trajectory_ids_kwargs_from_restart(restart_data):
    return {
        "global_MC_step_counter": restart_data["global_MC_step_counter"],
        "num_replicas": len(restart_data["cur_tps"]),
    }


def ordered_trajectory_from_restart(restart_data):
    return ordered_trajectory(
        restart_data["histogram"], **ordered_trajectory_ids_kwargs_from_restart(restart_data)
    )


def ordered_trajectory_ids_from_restart(restart_data):
    return ordered_trajectory_ids(
        restart_data["histogram"], **ordered_trajectory_ids_kwargs_from_restart(restart_data)
    )


def average_wait_number(histogram):
    return average_wait_number_from_traj_ids(ordered_trajectory_ids(histogram))


def average_wait_number_from_traj_ids(traj_ids):
    num_replicas = traj_ids.shape[-1]
    output = np.zeros((num_replicas,), dtype=int)
    cur_time = np.zeros((num_replicas,), dtype=int)
    prev_ids = traj_ids[0]
    for cur_ids in traj_ids[1:]:
        for counter, (cur_id, prev_id) in enumerate(zip(cur_ids, prev_ids)):
            if cur_id == prev_id:
                cur_time[counter] += 1
            else:
                cur_time[counter] = 0
        output[:] += cur_time[:]
        prev_ids = cur_ids

    return output / traj_ids.shape[0]
