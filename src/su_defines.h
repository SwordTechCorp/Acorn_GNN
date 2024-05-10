#ifndef __DEFINES_H_
#define __DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include <array>

//================================================//
// vector dimension defination
//================================================//
constexpr int   NODE_DIM = 12;
constexpr int   EDGE_DIM = 6; 
constexpr int   MLP_HIDDEN_DIM = 48;
constexpr int   EDGE_OUT_DIM = 1;
constexpr int   EDGE_BLOCK_IN_DIM = MLP_HIDDEN_DIM * 6;
constexpr int   EDGE_BLOCK_OUT_DIM = MLP_HIDDEN_DIM;
constexpr int   EDGE_BLOCK_END_DIM = 1; //E2

constexpr int   NODE_BLOCK_IN_DIM = MLP_HIDDEN_DIM * 4;
constexpr int   NODE_BLOCK_OUT_DIM = MLP_HIDDEN_DIM;
//================================================//


constexpr int   data_w = 8; //data width for attr_t

//================================================//
// data type 
//================================================//

typedef ap_fixed<8,4>  attr_t;
typedef ap_fixed<8,4>  weight_t;
typedef ap_uint<16>    index_t;

typedef std::array<attr_t,  EDGE_DIM>           edge;
typedef std::array<attr_t,  NODE_DIM>           node;
typedef std::array<attr_t,  MLP_HIDDEN_DIM>     hidden;
typedef std::array<attr_t,  MLP_HIDDEN_DIM*2>   hidden_2;
typedef std::array<attr_t,  MLP_HIDDEN_DIM*4>   hidden_4;
typedef std::array<attr_t,  MLP_HIDDEN_DIM*6>   hidden_6;
typedef std::array<attr_t,  1>                  edge_res;
typedef std::array<index_t, 2>                  idx_pair;

typedef attr_t DTYPE;
//================================================//
//const defined here
/*
constexpr int   MAX_NODE = 5;//739; //modify
constexpr int   MAX_EDGE = 10;//1252; //modify
constexpr int   SRC_NODE = 1; //2089;//202;
constexpr int   TOTAL_EDGE_PER_GRAPH = 10;//1252; //modify
constexpr int   MAX_NUM_GRAPH = 1;

constexpr int   MAX_NODE = 380;//739; //modify
constexpr int   MAX_EDGE = 634;//1252; //modify
constexpr int   SRC_NODE = 280;//202;
constexpr int   TOTAL_EDGE_PER_GRAPH = 634;//1252; //modify
constexpr int   MAX_NUM_GRAPH = 10;
*/

//================================================//
// graph size
//================================================//

constexpr int   MAX_NODE = 2500;//2500;//739; //modify
constexpr int   MAX_EDGE = 35000;//35000;//1252; //modify
constexpr int   SRC_NODE = 500; //2089;//202;
constexpr int   TOTAL_EDGE_PER_GRAPH = 35000;//35000;//1252; //modify
constexpr int   MAX_NUM_GRAPH = 1;


//================================================//
// message passing layers round
//================================================//
constexpr int   MP_LAYER = 8; 

//================================================//
// parallelism factor (E_BLOCK_PAR used)
// other may not used here
//================================================//
constexpr int   E_BLOCK_PAR = 4;
constexpr int   N_BLOCK_PAR = 1;
constexpr int   NS_BLOCK_PAR = 1;
constexpr int   MERGE_PAR = 6;


//================================================//
// parallelism factor for data in channel (no used here)
//================================================//
constexpr int   READ_W_PAR_PER_CH = 1; //number of channel for load weight
constexpr int   READ_N_PAR_PER_CH = 4; //number per channel for load node


//================================================//
// data type for AXI channel
//================================================//

typedef struct alignas(8 * E_BLOCK_PAR * data_w/8){
    attr_t e_attr[8 * E_BLOCK_PAR]; //edge_dim * E_PAR
}input_edge0;

/*
typedef struct alignas(READ_E_PAR_PER_CH1  * EDGE_DIM * data_w/8){
    attr_t e_attr[READ_E_PAR_PER_CH1  * EDGE_DIM]; //edge_dim * E_PAR
}input_edge1;
*/
typedef struct alignas(16 * READ_N_PAR_PER_CH *data_w/8){
    attr_t n_attr[16 * READ_N_PAR_PER_CH]; //edge_dim * E_PAR
    //attr_t dummy; //for 2^n bits per channel
}input_node;

typedef struct alignas(2 * E_BLOCK_PAR * 16/8){
    index_t i_attr[2 * E_BLOCK_PAR]; //edge_dim * E_PAR
}input_idx;


typedef struct alignas(1 * data_w/8){
    weight_t w_attr[1]; 
}input_weight;

typedef struct alignas(1 * E_BLOCK_PAR  * data_w/8){
    attr_t o_attr[1* E_BLOCK_PAR]; //edge_dim * E_PAR
}output_edge;


#endif
