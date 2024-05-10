# Acorn_GNN

Vitis HLS 2022.2 < br >
board: xcvu9p-flga2104-2-e < br >
testbench: su_test.cpp < br >
kernel: su_top_model.cpp < br >

Change path for input file:

1. src/su_host_load.cpp
   - #define DATA_DIR "/$your_path$/input_dataset/"
2. src/su_top_model.cpp
   - #include "/$your_path$/model_h/node_encoder_0_weight.h"  for all weights ROM

