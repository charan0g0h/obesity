gender_map = {
    'Female': 0,
    'Male': 1
}

yes_no_map = {
    'no': 0,
    'yes': 1
}

caec_map = {
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3
}

calc_map = {
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3
}

mtrans_map = {
    'Walking': 0,
    'Bike': 1,
    'Motorbike': 2,
    'Public_Transportation': 3,
    'Automobile': 4
}

obesity_map = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
reverse_map = {v: k for k, v in obesity_map.items()}