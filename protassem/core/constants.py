from numba.typed import Dict

atom_mass_dict = Dict()
atom_mass_dict["H"] = 1.008
atom_mass_dict["C"] = 12.011
atom_mass_dict["CA"] = 12.011
atom_mass_dict["N"] = 14.007
atom_mass_dict["O"] = 15.999
atom_mass_dict["P"] = 30.974
atom_mass_dict["S"] = 32.066

atomic_number_dict = Dict()
atomic_number_dict["H"] = 1.0
atomic_number_dict["C"] = 6.0
atomic_number_dict["CA"] = 6.0
atomic_number_dict["N"] = 7.0
atomic_number_dict["O"] = 8.0
atomic_number_dict["P"] = 15.0
atomic_number_dict["S"] = 16.0

VDW_RADII = {
    "H": 1.10,
    "C": 1.70,
    "CA": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
}

