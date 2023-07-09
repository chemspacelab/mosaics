# Possible valence values.
valences_int = {
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
    35: 1,
}
period_int = {
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

# Number of unshared pairs in valence shell.
unshared_pairs = {7: {3: 1}, 8: {2: 2}, 9: {1: 3}, 15: {3: 1}, 16: {2: 2}, 17: {1: 3}}

# Hybridization for a given "effective coordination number".

coord_num_hybrid = {4: "sp3", 3: "sp2", 2: "sp"}

max_ecn = 4

# Numbers of electrons of different type in valence shell.
s_int = {
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

p_int = {
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
