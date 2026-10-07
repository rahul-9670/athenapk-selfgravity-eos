# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

# Problem generators of this self-gravity + tabulated-EOS tree:
#   collapse_be   Bonnor-Ebert sphere collapse (self-gravity, tabulated EOS)
#   poisson_test  self-gravity multipole boundary condition, analytic gate
#   fofc_scalar   first-order flux correction x passive scalar consistency
#   grav_analytic, jeans_wave, grav_slab, polytrope, profile1d   self-gravity validation suite
#   eos_test      tabulated-EOS validation suite (waves, Riemann problems, compression, slabs)
target_sources(athenaPK PRIVATE
    collapse_be.cpp
    poisson_test.cpp
    fofc_scalar.cpp
    grav_analytic.cpp
    jeans_wave.cpp
    grav_slab.cpp
    polytrope.cpp
    profile1d.cpp
    eos_test.cpp
    )
