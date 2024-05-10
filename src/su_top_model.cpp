#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <array>
#include "hls_math.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"


#include "su_defines.h"
#include "su_util.cpp"

#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_encoder_0_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_encoder_3_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_encoder_6_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_encoder_0_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_encoder_3_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_encoder_6_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_encoder_0_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_encoder_3_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_encoder_6_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_encoder_0_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_encoder_3_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_encoder_6_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_network__0_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_network__3_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_network__6_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_network__0_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_network__3_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_network__6_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_network__0_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_network__3_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_network__6_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_network__0_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_network__3_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/node_network__6_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_decoder_0_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_decoder_3_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_decoder_6_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_decoder_0_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_decoder_3_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_decoder_6_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_output_transformer_0_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_output_transformer_1_weight.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_output_transformer_0_bias.h"
#include "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_h/edge_output_transformer_1_bias.h"



#define MAX_FIFO_DEPTH
//#define MIN_FIFO_DEPTH

using std::cout; 
using std::endl;  
using std::array;

void linear_dsp_4_12_48_small(
    hidden      input[4],     //input[4][48]
    edge_res    output[4],    //output[4][1]
    
    weight_t    weight[48],   //1D
    weight_t    bias[1]     //1D
)
{

#pragma HLS INLINE OFF

	hidden   out_el[4];
    hidden   temp[4];
    
#pragma HLS ARRAY_PARTITION variable=out_el complete
#pragma HLS ARRAY_PARTITION variable=temp complete 
	
	
    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL        
        for (int d_o = 0; d_o < 1; d_o++){
#pragma HLS UNROLL
            out_el[pe][d_o] =  bias[d_o];
        }
        
        for (int d_o = 0; d_o < 1; d_o++){
#pragma HLS UNROLL
            for (int d_i = 0; d_i < 48; d_i++){
#pragma HLS UNROLL
//#pragma HLS BIND_OP variable=temp op=mul impl=DSP latency=3

                //temp[pe][d_o] = input[pe][d_i] * weight[d_i][d_o];
                temp[pe][d_o] = product_dsp<attr_t,weight_t,attr_t>(input[pe][d_i], weight[d_i]);
                out_el[pe][d_o] += temp[pe][d_o];
            }
        }    
    
        for (int d_o = 0; d_o < 1; d_o++){
#pragma HLS UNROLL
            output[pe][d_o] = out_el[pe][d_o];
        }
    }
}
void linear_dsp_4_12_48(
    node        input[4],     //input[4][12]
    hidden      output[4],    //output[4][48]
    
    weight_t    weight[12][48],   //2D 
    weight_t    bias[48]     //1D
)
{

#pragma HLS INLINE OFF

	hidden   out_el[4];
    hidden   temp[4];
    
#pragma HLS ARRAY_PARTITION variable=out_el complete
#pragma HLS ARRAY_PARTITION variable=temp complete 
	
    //cout<<"in linear_dsp_4_12_48"<<endl;
	
    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL        
        for (int d_o = 0; d_o < 48; d_o++){
#pragma HLS UNROLL
            out_el[pe][d_o] =  0;//bias[d_o];
        }
        
        for (int d_o = 0; d_o < 48; d_o++){
#pragma HLS UNROLL
            for (int d_i = 0; d_i < 12; d_i++){
#pragma HLS UNROLL
//#pragma HLS BIND_OP variable=temp op=mul impl=DSP latency=3

                //temp[pe][d_o] = input[pe][d_i] * weight[d_i][d_o];
                temp[pe][d_o] = product_dsp<attr_t,weight_t,attr_t>(input[pe][d_i], weight[d_i][d_o]);
                out_el[pe][d_o] += temp[pe][d_o];
            }
        }    
    
        for (int d_o = 0; d_o < MLP_HIDDEN_DIM; d_o++){
#pragma HLS UNROLL
            output[pe][d_o] = out_el[pe][d_o];
        }
    }
}
//template<size_t DIM_IN, size_t DIM_OUT>
static void MLP_module_A(
    std::array<attr_t,  EDGE_BLOCK_IN_DIM>    *MLP_in,
    hidden   *MLP_out,
    
    weight_t buf_w00[24][1][12][48], //12*48
    weight_t buf_w01[4][1][12][48],
    weight_t buf_w02[4][1][12][48],
    weight_t buf_b00[48],
    weight_t buf_b01[48],
    weight_t buf_b02[48],
    int      IN_DIM
){
//#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
//#pragma HLS ARRAY_PARTITION variable=biases complete dim=1
//#pragma HLS PIPELINE

    //cout<<"into MLP_A"<<endl;
    //static int count=0;
    int BANK_N = ceildiv(IN_DIM, 12);
    const int BANK_H = ceildiv(MLP_HIDDEN_DIM, 12);
    
//MLP_L0=====================================================//

    hidden data0_logits[4];
#pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
            data0_logits[pe][dim] += buf_b00[dim];
        }
    }
    
    for(int bk=0; bk<BANK_N; bk++){
        
        node data_in[4];
        hidden tmp_out[4];
        
        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<12; dim++){
#pragma HLS UNROLL
                data_in[pe][dim] = MLP_in[pe][bk * 12 + dim];
            }
        }
        //cout<<"in l1 "<<bk<<endl;
        linear_dsp_4_12_48(
            data_in,
            tmp_out,
            buf_w00[bk][0],
            buf_b00
        );
        //cout<<"in l1 "<<bk<<endl;

        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<48; dim++){
#pragma HLS UNROLL
                data0_logits[pe][dim] += tmp_out[pe][dim];
            }    
        }
    }
    
    
    //cout<<"in lut "<<endl;
    hidden data0_norm[4];
#pragma HLS ARRAY_PARTITION variable=data0_norm complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        layer_norm<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_logits[pe], data0_norm[pe]);
        //layer_norm_LUT<attr_t, attr_t, ap_fixed<32, 16>,MLP_HIDDEN_DIM, 1024>(data0_logits[pe], data0_norm[pe]);
    }
    //cout<<"in lut "<<endl;
    
    
    hidden data0[4];
#pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        //relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_logits[pe], data0[pe]);
        relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_norm[pe], data0[pe]);
    }

//===========================================================//     
//MLP_L1=====================================================//

    //cout<<"MLP_L1_R: ";
    //print_fixed(data0, 12);

    hidden data1_logits[4];
#pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
            data1_logits[pe][dim] += buf_b01[dim];
        }
    }

    
    for(int bk=0; bk<BANK_H; bk++){
#pragma HLS loop_tripcount max = ceildiv(48, 12)
        node data_in[4];
        hidden tmp_out[4];
        
        
        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<12; dim++){
#pragma HLS UNROLL
                data_in[pe][dim] = data0[pe][bk * 12 + dim];
            }
        }
        
            
        linear_dsp_4_12_48(
            data_in,
            tmp_out,
            buf_w01[bk][0],
            buf_b01
        );

        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<48; dim++){
#pragma HLS UNROLL
                data1_logits[pe][dim] += tmp_out[pe][dim];
            }    
        }
    }
    
    hidden data1_norm[4];
#pragma HLS ARRAY_PARTITION variable=data1_norm complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        layer_norm<attr_t, attr_t, MLP_HIDDEN_DIM>(data1_logits[pe], data1_norm[pe]);
        //layer_norm_LUT<attr_t, attr_t, ap_fixed<32, 16>,MLP_HIDDEN_DIM, 1024>(data1_logits[pe], data1_norm[pe]);
    }
    
    hidden data1[4];
#pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data1_logits[pe], data1[pe]);
        //relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data1_norm[pe], data1[pe]);
    }

//===========================================================//     
//MLP_L2=====================================================//

    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
            MLP_out[pe][dim] += buf_b02[dim];
        }
    }
    for(int bk=0; bk<BANK_H; bk++){
#pragma HLS loop_tripcount max = ceildiv(48, 12)
        node data_in[4];
        hidden tmp_out[4];
        
        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<12; dim++){
#pragma HLS UNROLL
                data_in[pe][dim] = data1[pe][bk * 12 + dim];
            }
        }
            
        linear_dsp_4_12_48(
            data_in,
            tmp_out,
            buf_w01[bk][0],
            buf_b01
        );

        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<48; dim++){
#pragma HLS UNROLL
                MLP_out[pe][dim] += tmp_out[pe][dim];
            }    
        }
    }
    
    //count++;
}
//template<size_t DIM_IN, size_t DIM_OUT>
static void MLP_module_B(
    std::array<attr_t,  EDGE_BLOCK_IN_DIM>    *MLP_in,
    hidden   *MLP_out,
    
    weight_t buf_w00[24][1][12][48], //12*48
    weight_t buf_w01[4][1][12][48],
    weight_t buf_w02[4][1][12][48],
    weight_t buf_b00[48],
    weight_t buf_b01[48],
    weight_t buf_b02[48],
    int      IN_DIM
){
//#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
//#pragma HLS ARRAY_PARTITION variable=biases complete dim=1
//#pragma HLS PIPELINE


    int BANK_N = ceildiv(IN_DIM, 12);
    const int BANK_H = ceildiv(MLP_HIDDEN_DIM, 12);
    
//MLP_L0=====================================================//

    hidden data0_logits[4];
#pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
            data0_logits[pe][dim] += buf_b00[dim];
        }
    }
    
    for(int bk=0; bk<BANK_N; bk++){

        node data_in[4];
        hidden tmp_out[4];
        
        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<12; dim++){
#pragma HLS UNROLL
                data_in[pe][dim] = MLP_in[pe][bk * 12 + dim];
            }
        }
            
        linear_dsp_4_12_48(
            data_in,
            tmp_out,
            buf_w00[bk][0],
            buf_b00
        );

        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<48; dim++){
#pragma HLS UNROLL
                data0_logits[pe][dim] += tmp_out[pe][dim];
            }    
        }
    }
    
    hidden data0_norm[4];
#pragma HLS ARRAY_PARTITION variable=data0_norm complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        layer_norm<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_logits[pe], data0_norm[pe]);
        //layer_norm_LUT<attr_t, attr_t, ap_fixed<32, 16>,MLP_HIDDEN_DIM, 1024>(data0_logits[pe], data0_norm[pe]);
    }
    
    
    hidden data0[4];
#pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        //relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_logits[pe], data0[pe]);
        relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_norm[pe], data0[pe]);
    }

//===========================================================//     
//MLP_L1=====================================================//

    //cout<<"MLP_L1_R: ";
    //print_fixed(data0, 12);

    hidden data1_logits[4];
#pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
            data1_logits[pe][dim] += buf_b01[dim];
        }
    }

    
    for(int bk=0; bk<BANK_H; bk++){
#pragma HLS loop_tripcount max = ceildiv(48, 12)  
        node data_in[4];
        hidden tmp_out[4];
        
        
        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<12; dim++){
#pragma HLS UNROLL
                data_in[pe][dim] = data0[pe][bk * 12 + dim];
            }
        }
        
            
        linear_dsp_4_12_48(
            data_in,
            tmp_out,
            buf_w01[bk][0],
            buf_b01
        );

        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<48; dim++){
#pragma HLS UNROLL
                data1_logits[pe][dim] += tmp_out[pe][dim];
            }    
        }
    }
    
    hidden data1_norm[4];
#pragma HLS ARRAY_PARTITION variable=data1_norm complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        layer_norm<attr_t, attr_t, MLP_HIDDEN_DIM>(data1_logits[pe], data1_norm[pe]);
        //layer_norm_LUT<attr_t, attr_t, ap_fixed<32, 16>,MLP_HIDDEN_DIM, 1024>(data1_logits[pe], data1_norm[pe]);
    }
    
    hidden data1[4];
#pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        //relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data1_logits[pe], data1[pe]);
        relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data1_norm[pe], data1[pe]);
    }

//===========================================================//     
//MLP_L2=====================================================//

    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
            MLP_out[pe][dim] += buf_b02[dim];
        }
    }
    for(int bk=0; bk<BANK_H; bk++){
#pragma HLS loop_tripcount max = ceildiv(48, 12)
        node data_in[4];
        hidden tmp_out[4];
        
        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<12; dim++){
#pragma HLS UNROLL
                data_in[pe][dim] = data1[pe][bk * 12 + dim];
            }
        }
            
        linear_dsp_4_12_48(
            data_in,
            tmp_out,
            buf_w01[bk][0],
            buf_b01
        );

        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<48; dim++){
#pragma HLS UNROLL
                MLP_out[pe][dim] += tmp_out[pe][dim];
            }    
        }
    }
}

void edge_output_transform_kernal(
    std::array<attr_t, 1>  edge_transform_out[E_BLOCK_PAR],
    std::array<attr_t, MLP_HIDDEN_DIM>  edge_transform_in[E_BLOCK_PAR],
    
    weight_t buf_w50[ 4][1][12][48], //12*48
    weight_t buf_w51[48],

    weight_t buf_b50[48],
    weight_t buf_b51[1],
    
    int IN_DIM,
    int out_dim
){
//#pragma HLS PIPELINE
#pragma HLS INLINE
    
    int BANK_N = ceildiv(IN_DIM, 12);
    const int BANK_H = ceildiv(MLP_HIDDEN_DIM, 12);
    
//MLP_L0=====================================================//

    hidden data0_logits[4];
#pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
            data0_logits[pe][dim] += buf_b50[dim];
        }
    }
    
    for(int bk=0; bk<BANK_N; bk++){
        
        node data_in[4];
        hidden tmp_out[4];
        
        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<12; dim++){
#pragma HLS UNROLL
                data_in[pe][dim] = edge_transform_in[pe][bk * 12 + dim];
            }
        }
            
        linear_dsp_4_12_48(
            data_in,
            tmp_out,
            buf_w50[bk][0],
            buf_b50
        );

        for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<48; dim++){
#pragma HLS UNROLL
                data0_logits[pe][dim] += tmp_out[pe][dim];
            }    
        }
    }
    
    hidden data0_norm[4];
#pragma HLS ARRAY_PARTITION variable=data0_norm complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        layer_norm<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_logits[pe], data0_norm[pe]);
        //layer_norm_LUT<attr_t, attr_t, ap_fixed<32, 16>,MLP_HIDDEN_DIM, 1024>(data0_logits[pe], data0_norm[pe]);
    }
    
    
    hidden data0[4];
#pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    for(int pe=0; pe<4;pe++){
#pragma HLS UNROLL
        //relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_logits[pe], data0[pe]);
        relu<attr_t, attr_t, MLP_HIDDEN_DIM>(data0_norm[pe], data0[pe]);
    }
    
    //cout<<"MLP_L1_R: ";
    //print_fixed(data0, 12);

    //ATTR_T data1_logits[HIDDEN_DIM];  
    std::array<attr_t, 1> data1_logits[4];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    linear_dsp_4_12_48_small( 
        data0,
        edge_transform_out,
        buf_w51,
        buf_b51
    );
    
}

void write_back_new_algo(
	#ifdef __SYNTHESIS__
	attr_t                 edge_update[MAX_EDGE][MLP_HIDDEN_DIM],
	#else
	attr_t				   **edge_update,
	#endif
	output_edge            *O_ch0,
    
    weight_t buf_w40[ 4][1][12][48], //12*48
    weight_t buf_w41[ 4][1][12][48],
    weight_t buf_w42[ 4][1][12][48],
    weight_t buf_b40[48], 
    weight_t buf_b41[48],
    weight_t buf_b42[48],
    
    weight_t buf_w50[ 4][1][12][48], //12*48
    weight_t buf_w51[48],

    weight_t buf_b50[48],
    weight_t buf_b51[1],
    
    int num_of_edge
){
#pragma HLS INLINE
    int num_iters = ceildiv_even(num_of_edge, E_BLOCK_PAR);
	
    for(int iters = 0 ; iters < num_iters; iters++){
#pragma HLS PIPELINE OFF
#pragma HLS loop_tripcount max = ceildiv_even(MAX_EDGE, E_BLOCK_PAR)  
        
        hidden   edge_decoder_out[E_BLOCK_PAR];
        hidden_6 edge_decoder_in[E_BLOCK_PAR];
        edge_res edge_transform_out[E_BLOCK_PAR];
        output_edge out;

#pragma HLS ARRAY_PARTITION variable=edge_decoder_in    complete   dim=0 
#pragma HLS ARRAY_PARTITION variable=edge_decoder_out   complete   dim=0
#pragma HLS ARRAY_PARTITION variable=edge_transform_out complete   dim=0
        

        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL            
            if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL      
                    edge_decoder_in[pe][dim] = edge_update[iters * E_BLOCK_PAR + pe][dim];
                }
            }
            
        }
        
        
        MLP_module_A(
            edge_decoder_in,
            edge_decoder_out,
            
            buf_w40,
            buf_w41,
            buf_w42,
            buf_b40,
            buf_b41,
            buf_b42,
            
            MLP_HIDDEN_DIM
        );
        
        
        edge_output_transform_kernal(
            edge_transform_out,
            edge_decoder_out,
           
            buf_w50,
            buf_w51,
            buf_b50,
            buf_b51,
            
            MLP_HIDDEN_DIM,
            1
        );

        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL    
            if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                out.o_attr[pe] = edge_transform_out[pe][0];
            }
            
        }
        O_ch0[iters] = out;
    }
}

void node_network_new_algo(
    
    #ifdef __SYNTHESIS__
    attr_t  node_update[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
    
    attr_t  agg_src_map[MAX_NODE][MLP_HIDDEN_DIM],
    attr_t  agg_dst_map[MAX_NODE][MLP_HIDDEN_DIM],
    
    attr_t  node_input [MAX_NODE][MLP_HIDDEN_DIM],
    attr_t  node_static[MAX_NODE][MLP_HIDDEN_DIM],
    #else
    attr_t  ***node_update,
    attr_t  **agg_src_map,
    attr_t  **agg_dst_map,
    attr_t  **node_input,
    attr_t  **node_static,
    #endif
    
    
    
    weight_t buf_w30[16][1][12][48],    //48*4*48
    weight_t buf_w31[ 4][1][12][48],      //][2304],
    weight_t buf_w32[ 4][1][12][48],      //][2304],
    weight_t buf_b30[48],
    weight_t buf_b31[48],
    weight_t buf_b32[48],

    int num_of_edge
){ 
#pragma HLS INLINE
    int num_iters = ceildiv(MAX_NODE, E_BLOCK_PAR);
    for(int iters = 0; iters < num_iters; iters ++){
#pragma HLS PIPELINE OFF
#pragma HLS loop_tripcount max = ceildiv(MAX_NODE, E_BLOCK_PAR) 
        

        hidden_6  node_block_in[E_BLOCK_PAR];
        hidden   node_block_out[E_BLOCK_PAR];
        
#pragma HLS ARRAY_PARTITION variable=node_block_in  complete   dim=0 
#pragma HLS ARRAY_PARTITION variable=node_block_out complete   dim=0
         
        
        input_loop: for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL
            if(iters * E_BLOCK_PAR + pe < MAX_NODE){
                
                for(int dim = 0; dim < MLP_HIDDEN_DIM; dim ++){
#pragma HLS UNROLL
                    //cout<<"a2-"<<endl;
                    node_block_in[pe][dim]                      = agg_src_map[iters * E_BLOCK_PAR + pe][dim];
                    node_block_in[pe][MLP_HIDDEN_DIM * 1 + dim] = agg_dst_map[iters * E_BLOCK_PAR + pe][dim];
                    node_block_in[pe][MLP_HIDDEN_DIM * 2 + dim] = node_input [iters * E_BLOCK_PAR + pe][dim];
                    node_block_in[pe][MLP_HIDDEN_DIM * 3 + dim] = node_static[iters * E_BLOCK_PAR + pe][dim];
                }
            }
        }
        
        
        MLP_module_B(
            
            node_block_in,
            node_block_out,

            buf_w30,
            buf_w31,
            buf_w32,
            buf_b30,
            buf_b31,
            buf_b32,
        
            NODE_BLOCK_IN_DIM
        );
        

        output_loop: for(int cpy=0; cpy<E_BLOCK_PAR/2; cpy++){
#pragma HLS UNROLL
            for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL
                if(iters * E_BLOCK_PAR + pe < MAX_NODE){
                    for(int dim = 0; dim < MLP_HIDDEN_DIM; dim ++){
#pragma HLS UNROLL
                        node_update[cpy][iters * E_BLOCK_PAR + pe][dim] = node_block_out[pe][dim];
                    }
                }
            }
        }
    }
}
void aggregation_global_new_algo(
    
    //output to BRAM
    
    #ifdef __SYNTHESIS__
    attr_t agg_src_map[MAX_NODE][MLP_HIDDEN_DIM],
    attr_t agg_dst_map[MAX_NODE][MLP_HIDDEN_DIM],
    
    #else
    attr_t **agg_src_map,
    attr_t **agg_dst_map,
    #endif
    
    hls::stream<hidden>      edge_local_in[E_BLOCK_PAR],
    hls::stream<idx_pair>    idx_local_in[E_BLOCK_PAR],

    int num_of_edge
){
#pragma HLS INLINE
    int num_iters = ceildiv_even(num_of_edge, E_BLOCK_PAR);
    
    attr_t agg_src_map_tmp[E_BLOCK_PAR][MAX_NODE][MLP_HIDDEN_DIM];
    attr_t agg_dst_map_tmp[E_BLOCK_PAR][MAX_NODE][MLP_HIDDEN_DIM];
#pragma HLS ARRAY_PARTITION variable=agg_src_map_tmp  complete   dim=1  
#pragma HLS ARRAY_PARTITION variable=agg_dst_map_tmp  complete   dim=1

#pragma HLS ARRAY_PARTITION variable=agg_src_map_tmp  cyclic     dim=2  factor = E_BLOCK_PAR
#pragma HLS ARRAY_PARTITION variable=agg_dst_map_tmp  cyclic     dim=2  factor = E_BLOCK_PAR

#pragma HLS ARRAY_PARTITION variable=agg_src_map_tmp  complete   dim=3   
#pragma HLS ARRAY_PARTITION variable=agg_dst_map_tmp  complete   dim=3  

   
    for(int n=0; n<MAX_NODE; n++){
#pragma HLS UNROLL factor = E_BLOCK_PAR * 2
        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL   
            for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL   
                agg_dst_map_tmp[pe][n][dim] = 0;
                agg_src_map_tmp[pe][n][dim] = 0;
            }
        }
    }
	
    for(int iters = 0 ; iters < num_iters; iters++){
#pragma HLS PIPELINE 
#pragma HLS loop_tripcount max = ceildiv_even(MAX_EDGE, E_BLOCK_PAR) 
        
        //cout<<"agg iters: "<<iters<<endl;
        hidden edge_in[E_BLOCK_PAR];
        idx_pair   idx[E_BLOCK_PAR];

        
        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL  
            if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                edge_local_in[pe] >> edge_in[pe];
                idx_local_in[pe]  >> idx[pe];
                
                index_t s;
                index_t d;
                
                s = idx[pe][0];
                d = idx[pe][1];
                
                cout<<"agg iters "<<iters * E_BLOCK_PAR + pe <<" "<<s<<" "<<d<<endl;
                
                for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL  
                    agg_dst_map_tmp[pe][d][dim] += edge_in[pe][dim];
                    agg_src_map_tmp[pe][s][dim] += edge_in[pe][dim];
                }
            }
        }
    }
    
    cout<<"agg loop done"<<endl;
    
    for(int n=0; n<MAX_NODE; n++){
#pragma HLS UNROLL factor = E_BLOCK_PAR * 2
        hidden agg_dst;
        hidden agg_src;
        
        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL 
            for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL  
                if(pe==0){
                    agg_dst[dim] = agg_dst_map_tmp[pe][n][dim];
                    agg_src[dim] = agg_src_map_tmp[pe][n][dim];
                }
                else{
                    agg_dst[dim] += agg_dst_map_tmp[pe][n][dim];
                    agg_src[dim] += agg_src_map_tmp[pe][n][dim];
                }
                    
            }
        }
        
        for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL  
            agg_dst_map[n][dim] = agg_dst[dim];
            agg_src_map[n][dim] = agg_src[dim];
        }
        
    }
    
    cout<<"agg merage done"<<endl;
}
void edge_network_new_algo(
    
    hls::stream<hidden>      edge_embedding[E_BLOCK_PAR],
    hls::stream<idx_pair>    idx_stream[E_BLOCK_PAR],
    
    #ifdef __SYNTHESIS__
    //output BRAM
    attr_t          edge_update[MAX_EDGE][MLP_HIDDEN_DIM],
    
    attr_t          node_input[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
    attr_t          node_static[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
  
    //edge network input BRAM
    attr_t          edge_input[MAX_EDGE][MLP_HIDDEN_DIM],
    attr_t          edge_static[MAX_EDGE][MLP_HIDDEN_DIM],
    idx_pair        edge_idx[MAX_EDGE],
    
    
    #else
    //output BRAM  
    attr_t          **edge_update,
    
    attr_t          ***node_input,
    attr_t          ***node_static,

    //edge network input BRAM
    attr_t          **edge_input,
    attr_t          **edge_static,
    idx_pair        *edge_idx,
    
    #endif
    
    weight_t buf_w20[24][1][12][48],    //[13824], //48*6*48
    weight_t buf_w21[ 4][1][12][48],      //][2304],
    weight_t buf_w22[ 4][1][12][48],      //][2304],
    weight_t buf_b20[48], 
    weight_t buf_b21[48],
    weight_t buf_b22[48],
    
    int num_of_edge
){
#pragma HLS INLINE
    
	//cout<<"hihi9"<<endl;
    
    int num_iters = ceildiv_even(num_of_edge, E_BLOCK_PAR);
	
    for(int iters = 0 ; iters < num_iters; iters++){
#pragma HLS PIPELINE OFF
#pragma HLS loop_tripcount max = ceildiv_even(MAX_EDGE, E_BLOCK_PAR)  
        //cout<<"edge network loop "<<iters<<endl;
        
        
        hidden_6  edge_block_in[E_BLOCK_PAR];
        hidden    edge_block_out[E_BLOCK_PAR];
        

#pragma HLS ARRAY_PARTITION variable=edge_block_in  complete   dim=0 
#pragma HLS ARRAY_PARTITION variable=edge_block_out complete   dim=0
        
        input_loop: for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL
            idx_pair idx;
            index_t  s;
            index_t  d;
        
            if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                idx = edge_idx[iters * E_BLOCK_PAR + pe];
                s = idx[0];
                d = idx[1];
                                
                //cout<<"iters "<<iters * E_BLOCK_PAR + pe <<" "<<s<<" "<<d<<endl;
                //print_fixed(e, 8);
                for(int dim = 0; dim < MLP_HIDDEN_DIM; dim ++){
#pragma HLS UNROLL                
                    edge_block_in[pe][MLP_HIDDEN_DIM * 0 + dim] = edge_input[iters * E_BLOCK_PAR + pe][dim];
                    edge_block_in[pe][MLP_HIDDEN_DIM * 1 + dim] = edge_static[iters * E_BLOCK_PAR + pe][dim];
                    edge_block_in[pe][MLP_HIDDEN_DIM * 2 + dim] = node_input[pe%(E_BLOCK_PAR/2)][s][dim];       //random access
                    edge_block_in[pe][MLP_HIDDEN_DIM * 3 + dim] = node_static[pe%(E_BLOCK_PAR/2)][s][dim];  //random access
                    edge_block_in[pe][MLP_HIDDEN_DIM * 4 + dim] = node_input[pe%(E_BLOCK_PAR/2)][d][dim];       //random access
                    edge_block_in[pe][MLP_HIDDEN_DIM * 5 + dim] = node_static[pe%(E_BLOCK_PAR/2)][d][dim];  //random access
                }
                
                idx_stream[pe] << idx;
            }
        
        }
        
        MLP_module_A(
            edge_block_in,
            edge_block_out,
           
            buf_w20,
            buf_w21,
            buf_w22,
            buf_b20,
            buf_b21,
            buf_b22,
            MLP_HIDDEN_DIM
        );
		
		output_loop: for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL
            if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                edge_embedding[pe] << edge_block_out[pe];
                for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
                    edge_update[iters * E_BLOCK_PAR + pe][dim] = edge_block_out[pe][dim];
                }
            }
        }
    }
}           
void message_passing_new_algo(

    #ifdef __SYNTHESIS__
    //output BRAM
    attr_t          node_update[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
    attr_t          edge_update[MAX_EDGE][MLP_HIDDEN_DIM],
    
    attr_t          node_input [E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
    attr_t          node_static[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],

    //edge network input BRAM
    attr_t          edge_input[MAX_EDGE][MLP_HIDDEN_DIM],
    attr_t          edge_static[MAX_EDGE][MLP_HIDDEN_DIM],
    idx_pair        edge_idx[MAX_EDGE],
    
    #else
    //output BRAM  
    attr_t          ***node_update,
    attr_t          **edge_update,
    
    attr_t          ***node_input,
    attr_t          ***node_static,
    
    //edge network input BRAM
    attr_t          **edge_input,
    attr_t          **edge_static,
    idx_pair        *edge_idx,
    #endif

    
    weight_t buf_w20[24][1][12][48],    //[13824], //48*6*48
    weight_t buf_w21[ 4][1][12][48],      //][2304],
    weight_t buf_w22[ 4][1][12][48],      //][2304],
    weight_t buf_b20[48], 
    weight_t buf_b21[48],
    weight_t buf_b22[48],
    
    weight_t buf_w30[16][1][12][48],    //48*4*48
    weight_t buf_w31[ 4][1][12][48],      //][2304],
    weight_t buf_w32[ 4][1][12][48],      //][2304],
    weight_t buf_b30[48],
    weight_t buf_b31[48],
    weight_t buf_b32[48],
    
    int num_of_edge,
    int num_of_node
){
//#pragma HLS INLINE
//#pragma HLS INLINE off   
#pragma HLS DATAFLOW

    hls::stream<hidden>      edge_embedding[E_BLOCK_PAR];
    hls::stream<idx_pair>    idx[E_BLOCK_PAR];

#ifdef MIN_FIFO_DEPTH

#pragma HLS STREAM variable=edge_embedding		depth=2
#pragma HLS STREAM variable=idx       		    depth=2

#else

#pragma HLS STREAM variable=edge_embedding  depth=MAX_EDGE
#pragma HLS STREAM variable=idx             depth=MAX_EDGE

#endif 

#ifdef __SYNTHESIS__

    attr_t agg_src_map[MAX_NODE][MLP_HIDDEN_DIM];
    attr_t agg_dst_map[MAX_NODE][MLP_HIDDEN_DIM];
#else
    attr_t **agg_src_map = (attr_t **)malloc(MAX_NODE*sizeof(attr_t*));
	for(int i=0; i<MAX_NODE; i++){
        agg_src_map[i] = (attr_t *)malloc(MLP_HIDDEN_DIM * sizeof(attr_t));
    }
    
    attr_t **agg_dst_map = (attr_t **)malloc(MAX_NODE*sizeof(attr_t*));
	for(int i=0; i<MAX_NODE; i++){
        agg_dst_map[i] = (attr_t *)malloc(MLP_HIDDEN_DIM * sizeof(attr_t));
    }
#endif
    
#pragma HLS ARRAY_PARTITION variable=agg_src_map  cyclic    dim=1 factor = E_BLOCK_PAR
#pragma HLS ARRAY_PARTITION variable=agg_dst_map  cyclic    dim=1 factor = E_BLOCK_PAR  
#pragma HLS ARRAY_PARTITION variable=agg_src_map  complete  dim=2 
#pragma HLS ARRAY_PARTITION variable=agg_dst_map  complete  dim=2 


    edge_network_new_algo(
        edge_embedding,
        idx,
        
        edge_update,        //input - BRAM
        node_input,
        node_static,
        edge_input, 
        edge_static,
        edge_idx,     //output-streaming message

        
        buf_w20,
        buf_w21,
        buf_w22,
        buf_b20,
        buf_b21,
        buf_b22,
        
        num_of_edge
     );
	cout<<"edge_network_new_algo done"<<endl;
    
 
   //cout<<"node_block_fork done"<<endl;
//><><><<><>><><<><><><><><><><><><><>><
//================================================
//aggregation (many PE inside)
// original (critcal path II = #EDGE)
//================================================
   
    //cout<<"hihi9"<<endl;
    aggregation_global_new_algo(
        
        agg_src_map,
        agg_dst_map,
        
        edge_embedding,     //input-streaming message
        idx,
    
        num_of_edge
    );
    cout<<"aggregation_global_new_algo done"<<endl;
    //cout<<"hihi10"<<endl; 

   
    //inline!!
	node_network_new_algo(
        node_update,
        
        agg_src_map,
        agg_dst_map,
        node_input[0],
        node_static[0],
       
        buf_w30,
        buf_w31,
        buf_w32,
        buf_b30,
        buf_b31,
        buf_b32,
        
        num_of_edge
    );	
    cout<<"node_network_new_algo done"<<endl;
}
void load_graph_edge_new_algo(
 
    const input_edge0 *E_ch0_test,
    const input_idx   *I_ch0_test,
    
    #ifdef __SYNTHESIS__
    attr_t            edge_update[MAX_EDGE][MLP_HIDDEN_DIM],
    attr_t            edge_static[MAX_EDGE][MLP_HIDDEN_DIM],
    idx_pair          edge_idx[MAX_EDGE],
    #else
    attr_t            **edge_update,
    attr_t            **edge_static,
    idx_pair          *edge_idx,
    #endif
    
    weight_t buf_w10[1][1][12][48],  //[288]; //12*48
    weight_t buf_w11[4][1][12][48], //[2304],
    weight_t buf_w12[4][1][12][48], //[2304],
    weight_t buf_b10[48], //12*48
    weight_t buf_b11[48],
    weight_t buf_b12[48],
    
    int num_of_edge
){
#pragma HLS INLINE
//#pragma HLS INLINE off
//#pragma HLS PIPELINE II=1
    
    int num_iters = ceildiv_even(num_of_edge, E_BLOCK_PAR);
    
    for(int iters = 0 ; iters < num_iters; iters++){
#pragma HLS PIPELINE OFF   
#pragma HLS loop_tripcount max = ceildiv_even(MAX_EDGE, E_BLOCK_PAR)  
        input_edge0 ET0 = E_ch0_test[iters];
        input_idx   IT0 = I_ch0_test[iters];
//#pragma HLS array_partition variable=NT0 dim=1 complete       
 
        edge edge_encoder_in[E_BLOCK_PAR];
        hidden edge_encoder_out[E_BLOCK_PAR];
        idx_pair idx;
        
#pragma HLS array_partition variable=edge_encoder_in dim=1 complete
#pragma HLS array_partition variable=edge_encoder_in dim=2 complete    
#pragma HLS array_partition variable=edge_encoder_out dim=1 complete
#pragma HLS array_partition variable=edge_encoder_out dim=2 complete    

		
        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<EDGE_DIM; dim++){
#pragma HLS UNROLL
                if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                    edge_encoder_in[pe][dim] = ET0.e_attr[pe * EDGE_DIM + dim];
                    if(dim < 2){
                        idx[dim] = IT0.i_attr[pe * 2 + dim];
                    }
                }
            }
            if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                edge_idx[iters * E_BLOCK_PAR + pe] = idx;
            }
        }
		
        
        //padding
        //node padding_in[E_BLOCK_PAR];
        hidden_6 padding_in[READ_N_PAR_PER_CH];
        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL
             for(int dim=0; dim<EDGE_DIM; dim++){
#pragma HLS UNROLL                 
                padding_in[pe][dim] = edge_encoder_in[pe][dim];
                padding_in[pe][dim+6] = 0;
            }
        }
       
        
        //node encoder
        MLP_module_B(
            padding_in,
            edge_encoder_out,
           
            buf_w10,
            buf_w11,
            buf_w12,
            buf_b10,
            buf_b11,
            buf_b12,
            EDGE_DIM
        );
        
        //cout<<"node encoder out "<<iters<<endl;
        //print_fixed(node_encoder_out, 8);
        for(int pe=0; pe<E_BLOCK_PAR; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<MLP_HIDDEN_DIM; dim++){
#pragma HLS UNROLL
                if(iters * E_BLOCK_PAR + pe < MAX_EDGE){
                    edge_update[iters * E_BLOCK_PAR + pe][dim] = edge_encoder_out[pe][dim];
                    edge_static[iters * E_BLOCK_PAR + pe][dim] = edge_encoder_out[pe][dim];
                }
            }
        }
    }
}
void load_graph_node_new_algo(

    const input_node  *N_ch0_test,
    #ifdef __SYNTHESIS__
    attr_t          node_update[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
    attr_t          node_static[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],

    #else
    attr_t          ***node_update,
    attr_t          ***node_static,

    #endif
    
    weight_t buf_w00[1][1][12][48],//[576], //12*48
    weight_t buf_w01[4][1][12][48],//[2304],
    weight_t buf_w02[4][1][12][48],//[2304],
    weight_t buf_b00[48],
    weight_t buf_b01[48],
    weight_t buf_b02[48],
    
    int num_of_node
){
#pragma HLS INLINE
//#pragma HLS aggregate variable=N_ch0_test
    
    int num_iters = ceildiv_even(MAX_NODE, READ_N_PAR_PER_CH);
    
    
    for(int iters = 0 ; iters < num_iters; iters++){
#pragma HLS PIPELINE OFF 
#pragma HLS loop_tripcount max = ceildiv_even(MAX_NODE, READ_N_PAR_PER_CH)  
        input_node NT0 = N_ch0_test[iters];
//#pragma HLS array_partition variable=NT0 dim=1 complete       
 
        //node node_encoder_in[READ_N_PAR_PER_CH];
        hidden_6 node_encoder_in[READ_N_PAR_PER_CH];
        hidden node_encoder_out[READ_N_PAR_PER_CH];
#pragma HLS array_partition variable=node_encoder_in dim=1 complete
#pragma HLS array_partition variable=node_encoder_in dim=2 complete    
#pragma HLS array_partition variable=node_encoder_out dim=1 complete
#pragma HLS array_partition variable=node_encoder_out dim=2 complete    

		
        for(int pe=0; pe<READ_N_PAR_PER_CH; pe++){
#pragma HLS UNROLL
            for(int dim=0; dim<NODE_DIM; dim++){
#pragma HLS UNROLL
                if(iters * READ_N_PAR_PER_CH + pe < MAX_NODE){
                    node_encoder_in[pe][dim] = NT0.n_attr[pe * NODE_DIM + dim];
                }
            }
        }
		
        //node encoder
        MLP_module_A(
            node_encoder_in,
            node_encoder_out,
           
            buf_w00,
            buf_w01,
            buf_w02,
            buf_b00,
            buf_b01,
            buf_b02,
            NODE_DIM
        );
        
        //cout<<"node encoder out "<<iters<<endl;
        //print_fixed(node_encoder_out, 8);
        for(int cpy=0; cpy<E_BLOCK_PAR/2; cpy++){
#pragma HLS UNROLL
            for(int pe=0; pe<4; pe++){
#pragma HLS UNROLL
                for(int dim=0; dim<NODE_DIM; dim++){
#pragma HLS UNROLL
                    if(iters * READ_N_PAR_PER_CH + pe < MAX_NODE){
                        node_update[cpy][iters*READ_N_PAR_PER_CH+pe][dim] = node_encoder_out[pe][dim];
                        node_static[cpy][iters*READ_N_PAR_PER_CH+pe][dim] = node_encoder_out[pe][dim];
                    }
                }
            }
        }
    }
}
void pre_load_ping_pong_new_algo(

    //input
    const input_node  *N_ch0_test,
    const input_edge0 *E_ch0_test,
    const input_idx   *I_ch0_test,
    //const input_src   *S_ch0_test,
    
    
    #ifdef __SYNTHESIS__
    //output
    attr_t          node_update[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
    attr_t          node_static[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM],
  
    
    //edge network input BRAM
    attr_t          edge_update[MAX_EDGE][MLP_HIDDEN_DIM],
    attr_t          edge_static[MAX_EDGE][MLP_HIDDEN_DIM],
    idx_pair        edge_idx[MAX_EDGE],
    
    #else
        
    attr_t          ***node_update,
    attr_t          ***node_static,
    
    
    //edge network input BRAM
    attr_t          **edge_update,
    attr_t          **edge_static,
    idx_pair        *edge_idx,
    #endif

    
    weight_t buf_w00[1][1][12][48],//[576], //12*48
    weight_t buf_w01[4][1][12][48],//[2304],
    weight_t buf_w02[4][1][12][48],//[2304],
    weight_t buf_b00[48],
    weight_t buf_b01[48],
    weight_t buf_b02[48],

    weight_t buf_w10[1][1][12][48],  //[288]; //12*48
    weight_t buf_w11[4][1][12][48], //[2304],
    weight_t buf_w12[4][1][12][48], //[2304],
    weight_t buf_b10[48], //12*48
    weight_t buf_b11[48],
    weight_t buf_b12[48],
    
    //parameter
    int num_of_node,
    int num_of_edge
    
){
//#pragma HLS DATAFLOW
#pragma HLS INLINE


    load_graph_node_new_algo(
        
        N_ch0_test,        
        node_update, 
        node_static,
    
        buf_w00,
        buf_w01,
        buf_w02,
        buf_b00,
        buf_b01,
        buf_b02,
        
        num_of_node
    );
    cout<<"load_graph_node_L1 done"<<endl;
    
   
    load_graph_edge_new_algo(
        
        E_ch0_test,
        I_ch0_test,
        
        edge_update,
        edge_static, 
        edge_idx,
        
        buf_w10,
        buf_w11,
        buf_w12,
        buf_b10,
        buf_b11,
        buf_b12,
        
        num_of_edge
    );
    cout<<"load_graph_edge_v2 done"<<endl;
  
}
void load_weight_from_txt(
    weight_t buf_w00[1][1][12][48],//[576], //12*48
    weight_t buf_w01[4][1][12][48],//[2304],
    weight_t buf_w02[4][1][12][48],//[2304],
    weight_t buf_b00[48],
    weight_t buf_b01[48],
    weight_t buf_b02[48],

    weight_t buf_w10[1][1][12][48],  //[288]; //12*48
    weight_t buf_w11[4][1][12][48], //[2304],
    weight_t buf_w12[4][1][12][48], //[2304],
    weight_t buf_b10[48], //12*48
    weight_t buf_b11[48],
    weight_t buf_b12[48],
    
    //edge network
    weight_t buf_w20[8][24][1][12][48],    //[13824], //48*6*48
    weight_t buf_w21[8][ 4][1][12][48],      //][2304],
    weight_t buf_w22[8][ 4][1][12][48],      //][2304],
    weight_t buf_b20[8][48], 
    weight_t buf_b21[8][48],
    weight_t buf_b22[8][48],
    
    //node network
    weight_t buf_w30[8][16][1][12][48],    //48*4*48
    weight_t buf_w31[8][ 4][1][12][48],      //][2304],
    weight_t buf_w32[8][ 4][1][12][48],      //][2304],
    weight_t buf_b30[8][48],
    weight_t buf_b31[8][48],
    weight_t buf_b32[8][48],
    
    weight_t buf_w40[ 4][1][12][48], //12*48
    weight_t buf_w41[ 4][1][12][48],
    weight_t buf_w42[ 4][1][12][48],
    weight_t buf_b40[48], 
    weight_t buf_b41[48],
    weight_t buf_b42[48],
    
    weight_t buf_w50[ 4][1][12][48], //12*48
    weight_t buf_w51[48],

    weight_t buf_b50[48],
    weight_t buf_b51[1]
    
    
){
    //node encoder=================================================================================
    load_weights_from_txt_4d<weight_t, 1,1,12,48>(buf_w00, "node_encoder_0_weight.txt");
    load_weights_from_txt_4d<weight_t, 4,1,12,48>(buf_w01, "node_encoder_3_weight.txt");
    load_weights_from_txt_4d<weight_t, 4,1,12,48>(buf_w02, "node_encoder_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b00, "node_encoder_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b01, "node_encoder_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b02, "node_encoder_6_bias.txt");
    
    //edge encoder=================================================================================
    load_weights_from_txt_4d<weight_t, 1,1,12,48>(buf_w10, "edge_encoder_0_weight.txt");
    load_weights_from_txt_4d<weight_t, 4,1,12,48>(buf_w11, "edge_encoder_3_weight.txt");
    load_weights_from_txt_4d<weight_t, 4,1,12,48>(buf_w12, "edge_encoder_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b10, "edge_encoder_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b11, "edge_encoder_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b12, "edge_encoder_6_bias.txt");
    
    //edge network=================================================================================
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[0], "edge_network_0_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[0], "edge_network_0_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[0], "edge_network_0_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[0], "edge_network_0_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[0], "edge_network_0_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[0], "edge_network_0_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[1], "edge_network_1_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[1], "edge_network_1_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[1], "edge_network_1_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[1], "edge_network_1_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[1], "edge_network_1_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[1], "edge_network_1_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[2], "edge_network_2_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[2], "edge_network_2_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[2], "edge_network_2_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[2], "edge_network_2_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[2], "edge_network_2_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[2], "edge_network_2_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[3], "edge_network_3_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[3], "edge_network_3_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[3], "edge_network_3_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[3], "edge_network_3_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[3], "edge_network_3_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[3], "edge_network_3_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[4], "edge_network_4_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[4], "edge_network_4_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[4], "edge_network_4_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[4], "edge_network_4_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[4], "edge_network_4_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[4], "edge_network_4_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[5], "edge_network_5_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[5], "edge_network_5_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[5], "edge_network_5_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[5], "edge_network_5_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[5], "edge_network_5_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[5], "edge_network_5_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[6], "edge_network_6_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[6], "edge_network_6_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[6], "edge_network_6_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[6], "edge_network_6_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[6], "edge_network_6_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[6], "edge_network_6_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 24,1,12,48>(buf_w20[7], "edge_network_7_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w21[7], "edge_network_7_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w22[7], "edge_network_7_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b20[7], "edge_network_7_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b21[7], "edge_network_7_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b22[7], "edge_network_7_6_bias.txt");
    
    //node network=================================================================================
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[0], "node_network_0_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[0], "node_network_0_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[0], "node_network_0_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[0], "node_network_0_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[0], "node_network_0_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[0], "node_network_0_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[1], "node_network_1_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[1], "node_network_1_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[1], "node_network_1_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[1], "node_network_1_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[1], "node_network_1_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[1], "node_network_1_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[2], "node_network_2_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[2], "node_network_2_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[2], "node_network_2_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[2], "node_network_2_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[2], "node_network_2_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[2], "node_network_2_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[3], "node_network_3_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[3], "node_network_3_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[3], "node_network_3_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[3], "node_network_3_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[3], "node_network_3_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[3], "node_network_3_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[4], "node_network_4_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[4], "node_network_4_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[4], "node_network_4_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[4], "node_network_4_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[4], "node_network_4_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[4], "node_network_4_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[5], "node_network_5_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[5], "node_network_5_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[5], "node_network_5_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[5], "node_network_5_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[5], "node_network_5_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[5], "node_network_5_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[6], "node_network_6_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[6], "node_network_6_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[6], "node_network_6_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[6], "node_network_6_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[6], "node_network_6_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[6], "node_network_6_6_bias.txt");
    
    load_weights_from_txt_4d<weight_t, 16,1,12,48>(buf_w30[7], "node_network_7_0_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w31[7], "node_network_7_3_weight.txt");
    load_weights_from_txt_4d<weight_t,  4,1,12,48>(buf_w32[7], "node_network_7_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b30[7], "node_network_7_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b31[7], "node_network_7_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b32[7], "node_network_7_6_bias.txt");
    
    //edge decoder=================================================================================
    load_weights_from_txt_4d<weight_t, 4,1,12,48>(buf_w40, "edge_decoder_0_weight.txt");
    load_weights_from_txt_4d<weight_t, 4,1,12,48>(buf_w41, "edge_decoder_3_weight.txt");
    load_weights_from_txt_4d<weight_t, 4,1,12,48>(buf_w42, "edge_decoder_6_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b40, "edge_decoder_0_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b41, "edge_decoder_3_bias.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b42, "edge_decoder_6_bias.txt");
    
    //edge out transformer==========================================================================
    load_weights_from_txt_4d<weight_t, 1,1,12,48>(buf_w50, "edge_output_transform_0_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_w51, "edge_output_transform_3_weight.txt");
    load_weights_from_txt_1d<weight_t, 48>(buf_b50, "edge_output_transform_0_bias.txt");
    load_weights_from_txt_1d<weight_t,  1>(buf_b51, "edge_output_transform_3_bias.txt");
    
    
}
void interaction_network_new_algo(

    const input_node  *N_ch0_test,
    const input_edge0 *E_ch0_test,
    const input_idx   *I_ch0_test,
    //const input_src   *S_ch0_test,
    
    output_edge       *O3_ch0,
   
    //node encoder
    weight_t buf_w00[1][1][12][48],//[576], //12*48
    weight_t buf_w01[4][1][12][48],//[2304],
    weight_t buf_w02[4][1][12][48],//[2304],
    weight_t buf_b00[48],
    weight_t buf_b01[48],
    weight_t buf_b02[48],
    
    //edge encoder
    weight_t buf_w10[1][1][12][48],  //[288]; //12*48
    weight_t buf_w11[4][1][12][48], //[2304], //48*48
    weight_t buf_w12[4][1][12][48], //[2304], //48*48
    weight_t buf_b10[48], //12*48
    weight_t buf_b11[48],
    weight_t buf_b12[48],
    
    //edge network
    weight_t buf_w20[8][24][1][12][48],      //[13824], //48*6*48
    weight_t buf_w21[8][ 4][1][12][48],      //][2304], 
    weight_t buf_w22[8][ 4][1][12][48],      //][2304],
    weight_t buf_b20[8][48], 
    weight_t buf_b21[8][48],
    weight_t buf_b22[8][48],
    
    //node network
    weight_t buf_w30[8][16][1][12][48],    //48*4*48
    weight_t buf_w31[8][ 4][1][12][48],      //][2304],
    weight_t buf_w32[8][ 4][1][12][48],      //][2304],
    weight_t buf_b30[8][48],
    weight_t buf_b31[8][48],
    weight_t buf_b32[8][48],
    
    //edge decoder
    weight_t buf_w40[ 4][1][12][48], //12*48
    weight_t buf_w41[ 4][1][12][48],
    weight_t buf_w42[ 4][1][12][48],
    weight_t buf_b40[48], 
    weight_t buf_b41[48],
    weight_t buf_b42[48],
    
    weight_t buf_w50[ 4][1][12][48], //12*48
    weight_t buf_w51[48],

    weight_t buf_b50[48],
    weight_t buf_b51[1],
     
    
    int num_of_node,
    int *num_of_edges,
    int num_of_graph


){
	
	#ifdef __SYNTHESIS__
    //edge network input BRAM
    attr_t      node_update[2][E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM];
    attr_t      node_static[E_BLOCK_PAR/2][MAX_NODE][MLP_HIDDEN_DIM];
  
    //src list BRAM
    
    
    //edge network input BRAM
    attr_t      edge_update[2][MAX_EDGE][MLP_HIDDEN_DIM];
    attr_t      edge_static[MAX_EDGE][MLP_HIDDEN_DIM];
    idx_pair    edge_idx[MAX_EDGE];
    #else

    attr_t ****node_update = (attr_t ****)malloc(2*sizeof(attr_t***));
    for (int i=0; i<2; i++) {
        node_update[i] = (attr_t ***)malloc(E_BLOCK_PAR/2 * sizeof(attr_t**));
        for(int j=0; j<E_BLOCK_PAR/2; j++){
            node_update[i][j] = (attr_t **)malloc(MAX_NODE * sizeof(attr_t*));
            for (int k=0; k<MAX_NODE; k++) {
                node_update[i][j][k] = (attr_t *)malloc(MLP_HIDDEN_DIM * sizeof(attr_t));
            }
        }
    }
	
	
    attr_t ***node_static = (attr_t ***)malloc(E_BLOCK_PAR/2*sizeof(attr_t**));
	for(int i=0; i<E_BLOCK_PAR/2; i++){
        node_static[i] = (attr_t **)malloc(MAX_NODE * sizeof(attr_t*));
        for (int j=0; j<MAX_NODE; j++) {
            node_static[i][j] = (attr_t *)malloc(MLP_HIDDEN_DIM * sizeof(attr_t));
        }
    }

	
    
	attr_t ***edge_update = (attr_t ***)malloc(2*sizeof(attr_t**));
    for (int i = 0; i < 2; i++) {
        edge_update[i] = (attr_t **)malloc(MAX_EDGE * sizeof(attr_t*));
		for (int j = 0; j < MAX_EDGE; j++) {
			edge_update[i][j] = (attr_t *)malloc(MLP_HIDDEN_DIM * sizeof(attr_t));
		}
    }
	attr_t **edge_static = (attr_t **)malloc(MAX_EDGE*sizeof(attr_t*));
	for (int i = 0; i < MAX_EDGE; i++) {
        edge_static[i] = (attr_t *)malloc(MLP_HIDDEN_DIM * sizeof(attr_t));
    }
	
    idx_pair *edge_idx = (idx_pair *)malloc(MAX_EDGE*sizeof(idx_pair));
    #endif

/*      
#pragma HLS BIND_STORAGE variable=node_update   type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=node_static   type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=src_list      type=RAM_T2P impl=BRAM   

#pragma HLS BIND_STORAGE variable=edge_update   type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=edge_static   type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=edge_idx      type=RAM_T2P impl=BRAM
*/

//partition for pingpong
#pragma HLS ARRAY_PARTITION variable=node_update complete dim=1   
#pragma HLS ARRAY_PARTITION variable=edge_update complete dim=1  

//partition for parallelism cpy
#pragma HLS ARRAY_PARTITION variable=node_update  complete dim=2   
#pragma HLS ARRAY_PARTITION variable=node_static  complete dim=1   
   

//partition for parallelism 
#pragma HLS ARRAY_PARTITION variable=node_update  cyclic   dim=3  factor = READ_N_PAR_PER_CH
#pragma HLS ARRAY_PARTITION variable=node_static  cyclic   dim=2  factor = READ_N_PAR_PER_CH

#pragma HLS ARRAY_PARTITION variable=edge_update  cyclic   dim=2  factor = READ_N_PAR_PER_CH  
#pragma HLS ARRAY_PARTITION variable=edge_static  cyclic   dim=1  factor = READ_N_PAR_PER_CH  

//partition in hidden dimension
#pragma HLS ARRAY_PARTITION variable=node_update  complete dim=4  
#pragma HLS ARRAY_PARTITION variable=node_static  complete dim=3
#pragma HLS ARRAY_PARTITION variable=edge_update  complete dim=3    
#pragma HLS ARRAY_PARTITION variable=edge_static  complete dim=2 


    cout<<"hihi1"<<endl;
    
    //read graphs and encode nodes/edges
    pre_load_ping_pong_new_algo(

        //input
        &N_ch0_test[0],
        &E_ch0_test[0],
        &I_ch0_test[0],
        //&S_ch0_test[0],
        
        //output
        node_update[0],
        node_static,
        
        
        edge_update[0],
        edge_static,
        edge_idx,
         
        buf_w00,
        buf_w01,
        buf_w02,
        buf_b00,
        buf_b01,
        buf_b02,
        buf_w10,
        buf_w11,
        buf_w12,
        buf_b10,
        buf_b11,
        buf_b12,
        
        
        //parameter
        num_of_node,
        num_of_edges[0]
    );
	
    
    //run message-passing layers MP_LAYER times
    //cooperate with ping-pong buffer
    //two case for different read/write buffer
    
    for(int mp_loop=0; mp_loop < MP_LAYER; mp_loop ++){
#pragma HLS loop_tripcount max = MP_LAYER
        cout<<"mp_loop "<<mp_loop<<" ==============================="<<endl;
        if(mp_loop % 2 == 0){
            message_passing_new_algo(
                
                node_update[1],
                edge_update[1],
                
                node_update[0],
                node_static,
                


                edge_update[0],
                edge_static,
                edge_idx,
            
                buf_w20[mp_loop],
                buf_w21[mp_loop],
                buf_w22[mp_loop],
                buf_b20[mp_loop],
                buf_b21[mp_loop],
                buf_b22[mp_loop],
                buf_w30[mp_loop],
                buf_w31[mp_loop],
                buf_w32[mp_loop],
                buf_b30[mp_loop],
                buf_b31[mp_loop],
                buf_b32[mp_loop],

                num_of_edges[0],
                num_of_node
            );
        }else{
            message_passing_new_algo(
                
                node_update[0],
                edge_update[0],
                
                node_update[1],
                node_static,
                

                edge_update[1],
                edge_static,
                edge_idx,
                
                buf_w20[mp_loop],
                buf_w21[mp_loop],
                buf_w22[mp_loop],
                buf_b20[mp_loop],
                buf_b21[mp_loop],
                buf_b22[mp_loop],
                buf_w30[mp_loop],
                buf_w31[mp_loop],
                buf_w32[mp_loop],
                buf_b30[mp_loop],
                buf_b31[mp_loop],
                buf_b32[mp_loop],

                num_of_edges[0],
                num_of_node
            );
        }    
    }
    
    //edge decoder and output transformer
    write_back_new_algo(
        edge_update[0],
        &O3_ch0[0],
        
        buf_w40,
        buf_w41,
        buf_w42,
        buf_b40,
        buf_b41,
        buf_b42,
        buf_w50,
        buf_w51,
        buf_b50,
        buf_b51,
        
        num_of_edges[0]
    );
	
    
    
}

extern "C"{
void su_top_model(
    
  
    const input_node       N_ch0_test[MAX_NUM_GRAPH * ceildiv_even(MAX_NODE, READ_N_PAR_PER_CH)],
    const input_edge0      E_ch0_test[MAX_NUM_GRAPH * ceildiv_even(MAX_EDGE, E_BLOCK_PAR)],

    const input_idx        I_ch0_test[MAX_NUM_GRAPH * ceildiv_even(MAX_EDGE, E_BLOCK_PAR)],
    
	output_edge                O3_ch0[MAX_NUM_GRAPH * ceildiv_even(MAX_EDGE, E_BLOCK_PAR)],
    
    const int  num_graphs

){

#pragma HLS INTERFACE m_axi port = N_ch0_test  bundle = hbm0   max_widen_bitwidth=1024 max_read_burst_length=256  num_read_outstanding=16 depth=MAX_NUM_GRAPH * ceildiv_even(MAX_NODE, READ_N_PAR_PER_CH)  latency=64
#pragma HLS INTERFACE m_axi port = E_ch0_test  bundle = hbm1   max_widen_bitwidth=1024 max_read_burst_length=256  num_read_outstanding=16 depth=MAX_NUM_GRAPH * ceildiv_even(MAX_EDGE, E_BLOCK_PAR)  latency=64
#pragma HLS INTERFACE m_axi port = I_ch0_test  bundle = hbm15  max_widen_bitwidth=1024 max_read_burst_length=256  num_read_outstanding=16 depth=MAX_NUM_GRAPH * ceildiv_even(MAX_EDGE, E_BLOCK_PAR) latency=64
#pragma HLS INTERFACE m_axi port = O3_ch0      bundle = hbm12  max_widen_bitwidth=1024 max_write_burst_length=256 depth=MAX_NUM_GRAPH * ceildiv_even(MAX_EDGE, E_BLOCK_PAR) latency=64

#pragma HLS INTERFACE s_axilite         port = num_graphs                      bundle = control 
#pragma HLS INTERFACE s_axilite         port = return                          bundle = control     
 

#pragma HLS array_partition variable=buf_w00 dim=3 complete
#pragma HLS array_partition variable=buf_w00 dim=4 complete
#pragma HLS array_partition variable=buf_w01 dim=3 complete
#pragma HLS array_partition variable=buf_w01 dim=4 complete
#pragma HLS array_partition variable=buf_w02 dim=3 complete
#pragma HLS array_partition variable=buf_w02 dim=4 complete

#pragma HLS array_partition variable=buf_b00 dim=1 complete
#pragma HLS array_partition variable=buf_b01 dim=1 complete
#pragma HLS array_partition variable=buf_b02 dim=1 complete

/* #pragma HLS BIND_STORAGE variable=buf_w00   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w01   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w02   type=ROM_NP impl=LUTRAM

#pragma HLS BIND_STORAGE variable=buf_b00   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b01   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b02   type=ROM_NP impl=LUTRAM */
    

    
    
#pragma HLS array_partition variable=buf_w10 dim=3 complete
#pragma HLS array_partition variable=buf_w10 dim=4 complete
#pragma HLS array_partition variable=buf_w11 dim=3 complete
#pragma HLS array_partition variable=buf_w11 dim=4 complete
#pragma HLS array_partition variable=buf_w12 dim=3 complete
#pragma HLS array_partition variable=buf_w12 dim=4 complete

#pragma HLS array_partition variable=buf_b10 dim=1 complete
#pragma HLS array_partition variable=buf_b11 dim=1 complete
#pragma HLS array_partition variable=buf_b12 dim=1 complete
/* 
#pragma HLS BIND_STORAGE variable=buf_w10   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w11   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w12   type=ROM_NP impl=LUTRAM

#pragma HLS BIND_STORAGE variable=buf_b10   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b11   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b12   type=ROM_NP impl=LUTRAM */


    
    
#pragma HLS array_partition variable=buf_w20 dim=4 complete
#pragma HLS array_partition variable=buf_w20 dim=5 complete
#pragma HLS array_partition variable=buf_w21 dim=4 complete
#pragma HLS array_partition variable=buf_w21 dim=5 complete
#pragma HLS array_partition variable=buf_w22 dim=4 complete
#pragma HLS array_partition variable=buf_w22 dim=5 complete

#pragma HLS array_partition variable=buf_b20 dim=2 complete
#pragma HLS array_partition variable=buf_b21 dim=2 complete
#pragma HLS array_partition variable=buf_b22 dim=2 complete
/* 
#pragma HLS BIND_STORAGE variable=buf_w20   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w21   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w22   type=ROM_NP impl=LUTRAM

#pragma HLS BIND_STORAGE variable=buf_b20   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b21   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b22   type=ROM_NP impl=LUTRAM */
    

#pragma HLS array_partition variable=buf_w30 dim=4 complete
#pragma HLS array_partition variable=buf_w30 dim=5 complete
#pragma HLS array_partition variable=buf_w31 dim=4 complete
#pragma HLS array_partition variable=buf_w31 dim=5 complete
#pragma HLS array_partition variable=buf_w32 dim=4 complete
#pragma HLS array_partition variable=buf_w32 dim=5 complete

#pragma HLS array_partition variable=buf_b30 dim=2 complete
#pragma HLS array_partition variable=buf_b31 dim=2 complete
#pragma HLS array_partition variable=buf_b32 dim=2 complete
/* 
#pragma HLS BIND_STORAGE variable=buf_w30   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w31   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w32   type=ROM_NP impl=LUTRAM

#pragma HLS BIND_STORAGE variable=buf_b30   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b31   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b32   type=ROM_NP impl=LUTRAM */

  

#pragma HLS array_partition variable=buf_w40 dim=3 complete
#pragma HLS array_partition variable=buf_w40 dim=4 complete
#pragma HLS array_partition variable=buf_w41 dim=3 complete
#pragma HLS array_partition variable=buf_w41 dim=4 complete
#pragma HLS array_partition variable=buf_w42 dim=3 complete
#pragma HLS array_partition variable=buf_w42 dim=4 complete

#pragma HLS array_partition variable=buf_b40 dim=1 complete
#pragma HLS array_partition variable=buf_b41 dim=1 complete
#pragma HLS array_partition variable=buf_b42 dim=1 complete
/* 
#pragma HLS BIND_STORAGE variable=buf_w40   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w41   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w42   type=ROM_NP impl=LUTRAM

#pragma HLS BIND_STORAGE variable=buf_b40   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b41   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b42   type=ROM_NP impl=LUTRAM */
    


#pragma HLS array_partition variable=buf_w50 dim=3 complete
#pragma HLS array_partition variable=buf_w50 dim=4 complete

#pragma HLS array_partition variable=buf_w51 dim=1 complete
#pragma HLS array_partition variable=buf_b50 dim=1 complete
#pragma HLS array_partition variable=buf_b51 dim=1 complete
   
/* #pragma HLS array_partition variable=buf_w50 block factor = 12 //dim=1 complete
#pragma HLS array_partition variable=buf_w51 block factor = 12 //dim=1 complete

#pragma HLS array_partition variable=buf_b50 dim=1 complete
#pragma HLS array_partition variable=buf_b51 dim=1 complete


#pragma HLS BIND_STORAGE variable=buf_w50   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_w51   type=ROM_NP impl=LUTRAM

#pragma HLS BIND_STORAGE variable=buf_b50   type=ROM_NP impl=LUTRAM
#pragma HLS BIND_STORAGE variable=buf_b51   type=ROM_NP impl=LUTRAM */

/*
#ifndef __SYNTHESIS__
    load_weight_from_txt(
        buf_w00,
        buf_w01,
        buf_w02,
        buf_b00,
        buf_b01,
        buf_b02,
        buf_w10,
        buf_w11,
        buf_w12,
        buf_b10,
        buf_b11,
        buf_b12,
        
        buf_w20,
        buf_w21,
        buf_w22,
        buf_b20,
        buf_b21,
        buf_b22,
        buf_w30,
        buf_w31,
        buf_w32,
        buf_b30,
        buf_b31,
        buf_b32,
        
        buf_w40,
        buf_w41,
        buf_w42,
        buf_b40,
        buf_b41,
        buf_b42,
        buf_w50,
        buf_w51,
        buf_b50,
        buf_b51
        
    );
#endif
*/
 
   
    int num_of_edges[MAX_NUM_GRAPH];
  
    
    int num_of_node = MAX_NODE;

    //int num_of_graph = 1;
    
    int loop_per_graph = num_graphs;//TOTAL_EDGE_PER_GRAPH / MAX_EDGE + 1;
    for(int i=0; i<MAX_NUM_GRAPH; i++){
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount max = MAX_NUM_GRAPH
        num_of_edges[i] = MAX_EDGE;
    }
    
    //main kernel
	interaction_network_new_algo(
		
		N_ch0_test,
		E_ch0_test,
		I_ch0_test,
        //S_ch0_test,
		O3_ch0,
        
        buf_w00,
        buf_w01,
        buf_w02,
        buf_b00,
        buf_b01,
        buf_b02,
        buf_w10,
        buf_w11,
        buf_w12,
        buf_b10,
        buf_b11,
        buf_b12,
        
        buf_w20,
        buf_w21,
        buf_w22,
        buf_b20,
        buf_b21,
        buf_b22,
        buf_w30,
        buf_w31,
        buf_w32,
        buf_b30,
        buf_b31,
        buf_b32,
        
        buf_w40,
        buf_w41,
        buf_w42,
        buf_b40,
        buf_b41,
        buf_b42,
        buf_w50,
        buf_w51,
        buf_b50,
        buf_b51,
        
		num_of_node,
		num_of_edges,
        loop_per_graph
	);
	
    cout<<"hihi_end"<<endl;
    
}
}
