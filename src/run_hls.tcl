open_project try

# Add design files
add_files su_top_model.cpp
# Add test bench & files
add_files -tb su_test.cpp

# Set the top-level function
set_top su_top_model

# ########################################################
# Create a solution
open_solution -reset solution1 -flow_target vitis
# Define technology and clock rate
set_part  {xcvu9p-flga2104-2-e}
create_clock -period 10

# Set variable to select which steps to execute
set hls_exec 2

csim_design
# Set any optimization directives
# End of directives

if {$hls_exec == 1} {
	# Run Synthesis and Exit
	csynth_design
} elseif {$hls_exec == 2} {
	# Run Synthesis, RTL Simulation and Exit
	csynth_design
	cosim_design
} elseif {$hls_exec == 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation and Exit
	csynth_design
	cosim_design
	export_design -flow syn 
} else {
	# Default is to exit after running csynth
	csynth_design
}

exit