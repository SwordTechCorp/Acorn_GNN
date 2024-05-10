# Acorn_GNN

Vitis HLS 2022.2 <br>
board: xcvu9p-flga2104-2-e <br>
testbench: su_test.cpp <br>
kernel: su_top_model.cpp <br>

> [!IMPORTANT]
> The temp version doesn't check for functional correctness, i.e., the answer might be wrong.

## Change path for input file:

1. src/su_host_load.cpp
   - #define DATA_DIR "/$your_path$/input_dataset/"
2. src/su_top_model.cpp
   - #include "/$your_path$/model_h/node_encoder_0_weight.h"  for all weights ROM

## Run vitis_hls with command line, type: <be>

```bash
cd src
vitis_hls -f run_hls.tcl
```

flow <be>
- C-sim: simulate with cpp file, without HLS pragma
- C-synth: compile cpp to Verilog/VHDL
- Co-sim: simulate with Verilog/VHDL file
- RTL-synth: synthesize the Verilog code

> [!NOTE]
> All steps take hours to run.
