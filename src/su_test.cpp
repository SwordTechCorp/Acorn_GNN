//#include "xcl2.hpp"
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <ap_int.h>
#include <fstream>
#include <chrono>



#include "su_host_defines.h"
#include "su_host_load.h"
//#include "su_defines.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;
using namespace std;


extern "C"{
void su_top_model(
    
    const input_node       N_ch0_test[MAX_NUM_GRAPH_HOST * ceildiv(MAX_NODE_HOST, N_PAR)],
    const input_edge0      E_ch0_test[MAX_NUM_GRAPH_HOST * ceildiv_even(MAX_EDGE_HOST, E_PAR)],
    const input_idx        I_ch0_test[MAX_NUM_GRAPH_HOST * ceildiv_even(MAX_EDGE_HOST, E_PAR)],
 
	output_edge                O3_ch0[MAX_NUM_GRAPH_HOST * ceildiv_even(MAX_EDGE_HOST, E_PAR)],
    
    const int               num_graphs
    
);
}

int main(int argc, char **argv) {
	int num_of_graph = MAX_NUM_GRAPH_HOST;
   

	bool **golden_e_fpga = (bool **) malloc(sizeof(bool *) * MAX_NUM_GRAPH_HOST);
    for (int i = 0; i < MAX_NUM_GRAPH_HOST; i++) {
        golden_e_fpga[i] = (bool *) malloc(MAX_TOTAL_EDGE_HOST * sizeof(bool));
    }
    
    
    //vector<DTYPE >  out_e_fpga(MAX_NUM_GRAPH_HOST * MAX_TOTAL_EDGE_HOST * 1);
    //vector<output_edge>  out_e_fpga_hbm(MAX_NUM_GRAPH_HOST * (MAX_TOTAL_EDGE_HOST / MAX_EDGE_HOST +1) * ((MAX_EDGE_HOST/E_PAR)+1));
   
    
    int num_graphs = MAX_NUM_GRAPH_HOST;
    
    //simple test=========================//
    input_node  *NT_S  = (input_node *)  malloc(sizeof(input_node)  * MAX_NUM_GRAPH_HOST * ceildiv(MAX_NODE_HOST, N_PAR));
	input_edge0 *ET_S  = (input_edge0 *) malloc(sizeof(input_edge0) * MAX_NUM_GRAPH_HOST * ceildiv_even(MAX_EDGE_HOST, E_PAR));
    input_idx   *IT_S  = (input_idx *)   malloc(sizeof(input_idx)   * MAX_NUM_GRAPH_HOST * ceildiv_even(MAX_EDGE_HOST, E_PAR));
	output_edge *OT_S  = (output_edge *) malloc(sizeof(output_edge) * MAX_NUM_GRAPH_HOST * ceildiv_even(MAX_EDGE_HOST, E_PAR));
	
    
    load_data_host_v2(
        num_of_graph, 
        NT_S,
        ET_S,
        IT_S,
        golden_e_fpga
    );
    
    cout<<"end load_data_host"<<endl;
 
    //simple test=========================//
    cout<<"load node host"<<endl;
    for(int i=0; i<3; i++){
        for(int par=0; par < N_PAR; par++){
            for(int dim=0; dim<NODE_DIM_HOST; dim++){
                cout<<NT_S[i].n_attr[par * NODE_DIM_HOST + dim]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    
    cout<<"load edge host"<<endl;
    for(int i=0; i<3; i++){
        for(int par=0; par < E_PAR; par++){
            for(int dim=0; dim<EDGE_DIM_HOST; dim++){
                cout<<ET_S[i].e_attr[par * EDGE_DIM_HOST + dim]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    
    cout<<"load index host"<<endl;
    for(int i=0; i<3; i++){
        for(int par=0; par < E_PAR; par++){
            for(int dim=0; dim<2; dim++){
                cout<<IT_S[i].i_attr[par * 2 + dim]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    
    
	su_top_model(
    
        NT_S,
        ET_S,
        IT_S,

		
		OT_S,
        num_graphs
	);
	
    
    //================================================================//
	//check golden here                                               //
    //================================================================//
    /*
	for(int g=0; g<num_graphs; g++){
		//bool out_data[MAX_TOTAL_EDGE_HOST];
		int mismatch_cnt = 0;
		cout<<"check ans graph "<<g<<"======="<<endl;
		for(int i=0; i< MAX_TOTAL_EDGE_HOST; i++){
			
			
			if(OT[g*ceildiv(MAX_TOTAL_EDGE_HOST, E2_PAR)+i/E2_PAR].o_attr[i%E2_PAR] - golden_e_fpga[g][i] > 0.5 ||
			 golden_e_fpga[g][i] - OT[g*ceildiv(MAX_TOTAL_EDGE_HOST, E2_PAR)+i/E2_PAR].o_attr[i%E2_PAR] > 0.5){
				 if(i_fpga_f[g*(MAX_TOTAL_EDGE_HOST*2)+2*i]!=i_fpga_f[g*(MAX_TOTAL_EDGE_HOST*2)+2*i+1]){
					 mismatch_cnt++;
					
					if(g==0){
						std::cout<<"error "<<i_fpga_f[g*(MAX_TOTAL_EDGE_HOST*2) + 2*i]<<" to "<<i_fpga_f[g*(MAX_TOTAL_EDGE_HOST*2) + 2*i+1]<<std::endl;
					}
				}
			 }
		
		} 
		cout<<"mismatch_count: "<<mismatch_cnt<<endl;
	}
    */
    //================================================================//
    

    free(NT_S);
	free(ET_S);
	free(IT_S);

	free(OT_S);
	for (int i = 0; i < MAX_NUM_GRAPH_HOST; i++) {
        free(golden_e_fpga[i]);
    }
	free(golden_e_fpga);

	return 0;
}
