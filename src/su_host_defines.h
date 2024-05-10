#ifndef HOST_DEFINES
#define HOST_DEFINES
#include <vector>
#include <cstdlib>
#include <ap_int.h>
typedef ap_fixed<8,4> DTYPE;
typedef ap_uint<16>    index_t;

const int data_w = 8;
//typedef ap_fixed<14,7> DTYPE;


/*
const int MAX_NUM_GRAPH_HOST = 1;
const int MAX_NODE_HOST = 5;
const int MAX_EDGE_HOST = 10;
const int MAX_TOTAL_EDGE_HOST = 10;
const int MAX_SRC_HOST = 1;

const int MAX_NUM_GRAPH_HOST = 1;
const int MAX_NODE_HOST = 738;
const int MAX_EDGE_HOST = 1252;
const int MAX_TOTAL_EDGE_HOST = 1252;
const int MAX_SRC_HOST = 202;
*/
const int MAX_NUM_GRAPH_HOST = 1;
const int MAX_NODE_HOST = 2500;
const int MAX_EDGE_HOST = 35000;
const int MAX_TOTAL_EDGE_HOST = 35000;
const int MAX_SRC_HOST = 500;

const int MLP_LAYER_HOST = 3;
const int MAX_WEIGHT_DIM_HOST = 48*48*6;//80;
const int MAX_BIAS_DIM_HOST= 48;


const int NODE_DIM_HOST = 12;
const int EDGE_DIM_HOST = 6;

const int E_PAR = 4;
const int E_ch0_PAR = 1;
const int E_ch1_PAR = 0;
const int N_PAR = 4;
const int W_PAR = 1;

typedef struct alignas(8 * E_PAR * 8/8){
    DTYPE e_attr[8 * E_PAR]; //edge_dim * E_PAR
}input_edge0;

/*
typedef struct alignas(EDGE_DIM_HOST * E_ch1_PAR * 16/8){
    DTYPE e_attr[EDGE_DIM_HOST * E_ch1_PAR]; //edge_dim * E_PAR
}input_edge1;
*/

typedef struct alignas(16 * N_PAR * 8/8){
    DTYPE n_attr[16 * N_PAR]; //read node
    //DTYPE dummy;
}input_node;

typedef struct alignas(2 * E_PAR * 16/8){
    ap_uint<16> i_attr[2 * E_PAR]; //edge_dim * E_PAR
}input_idx;

typedef struct alignas(N_PAR * 8/8){
    ap_uint<16> s_attr[N_PAR]; //edge_dim * E_PAR
}input_src;

typedef struct alignas(W_PAR * 8/8){
    DTYPE w_attr[W_PAR]; //edge_dim * E_PAR
}input_weight;

typedef struct alignas(1 * E_PAR * 8/8){
    DTYPE o_attr[1 * E_PAR]; //edge_dim * E_PAR
}output_edge;






#endif
