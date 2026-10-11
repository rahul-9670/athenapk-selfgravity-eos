# Copyright (c) 2020-2021, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    athenaPK
        main.cpp
        eos/adiabatic_glmmhd.cpp
        units.hpp
        units/physical_units.hpp
        bvals/boundary_conditions_apk.hpp
        bvals/fused_bc_mesh.cpp
        diagnostics/cons_diag.cpp
        diagnostics/cons_diag.hpp
        diagnostics/grav_diag.cpp
        diagnostics/grav_diag.hpp
        hydro/double_flux.cpp
        hydro/dual_flux_density.cpp
        hydro/dual_double_flux.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        hydro/glmmhd/dedner_source.cpp
        hydro/prolongation/custom_ops.hpp
        refinement/gradient.cpp
        refinement/other.cpp
        refinement/jeans.cpp
        tracers/tracers.cpp
        tracers/tracers.hpp
        utils/few_modes_ft.cpp
        utils/few_modes_ft.hpp
        self_gravity/self_gravity.cpp
        self_gravity/self_gravity_driver.cpp
        radiation/radiation.cpp
        radiation/radiation_sn.cpp
        radiation/radiation_sn_conserve.cpp
        radiation/radiation_sn_testsrc.cpp
        radiation/radiation_sn_criterion.cpp
        radiation/radiation_sn_fc.cpp
        radiation/radiation_sn_soulim.cpp
        radiation/radiation_sn_precond.cpp
        radiation/radiation_sn_wavefront.cpp
        radiation/radiation_sn_fastops.cpp
        radiation/radiation_sn_graydsa.cpp
        radiation/radiation_tasks.cpp
)

add_subdirectory(pgen)

target_link_libraries(athenaPK PRIVATE parthenon)
