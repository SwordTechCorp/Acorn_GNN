# Acorn_GNN

Vitis HLS 2022.2
board: xcvu9p-flga2104-2-e
testbench: su_test.cpp
kernel: su_top_model.cpp

Change path for input file:
src/su_host_load.cpp
- #define DATA_DIR "/$your_path$/input_dataset/"
src/su_top_model.cpp
- #include "/$your_path$/model_h/node_encoder_0_weight.h"  for all weights ROM

