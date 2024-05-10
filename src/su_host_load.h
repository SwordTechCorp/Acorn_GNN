#include "su_host_defines.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;
#ifndef HOST_LOADER
#define HOST_LOADER

#define WEIGHTS_DIR "/home/suyuru/myMP/mymy_v2/src/weights"
//#define DATA_DIR "/home/suyuru/myMP/mymy_v2/src/data/phi1_modulized/"
#define DATA_DIR "/home/suyuru/git_code/acorn_qat/acorn/input_dataset/"
//#define DATA_DIR "/home/suyuru/myMP/regen_input/phi1_modulized_pt_05_10000_97000/"
//#define DATA_DIR "/home/suyuru/myMP/regen_input/phi2_modulized_pt_1_380_634/"
constexpr int ceildiv_even(int dividend, int divisor)
{
	int out = ((dividend + divisor - 1) / divisor);
	if(out% 2 ==1)
		return out + 1;
    else return out;
}
constexpr int ceildiv(int dividend, int divisor)
{
    return (dividend + divisor - 1) / divisor;
}

void load_data_host_v2(
    int num_of_graph,
    input_node  *NT_S,
	input_edge0 *ET_S,
    input_idx   *IT_S,
    bool **golden_e_fpga
){
    
    for(int g=0; g<num_of_graph;g++){
        
        cout<<"start"<<endl;
		
        std::string in_node_data;  
        in_node_data = std::string(DATA_DIR) + "node_feature_"+std::to_string(g)+".txt";
        std::ifstream fin_n(in_node_data);
        
        std::string in_edge_data;  
        in_edge_data = std::string(DATA_DIR) + "edge_feature_"+std::to_string(g)+".txt";
        std::ifstream fin_e(in_edge_data);
        
        std::string in_idx_data;  
        in_idx_data = std::string(DATA_DIR) + "edge_index_"+std::to_string(g)+".txt";
        std::ifstream fin_i(in_idx_data);
        
        cout<<"b"<<endl;
        /*
        std::string golden_data;  
        golden_data = std::string(DATA_DIR) + "post_matA_gold"+std::to_string(g)+".dat";
        std::ifstream fgold(golden_data);  
        
        std::string out_data;  
        out_data = std::string(DATA_DIR) + "predict_graph"+std::to_string(g)+".dat";
        std::ofstream fout(out_data);   
        */
               
        std::string iline;
		
		
        if(fin_n.is_open()){
            
			std::cout<<"load with "<<in_node_data<<" \n";
            std::vector<float> in;
            for (int r = 0; r < MAX_NODE_HOST; r++){
                for (int c = 0; c < NODE_DIM_HOST; c++){
                    float f;
                    fin_n >> f;  // Take input from file and put into myArray
                    in.push_back(f);
                }
            }
            
            for(int i=0; i<MAX_NODE_HOST; i++){
                for(int dim=0; dim<NODE_DIM_HOST; dim++){
                    NT_S[i/N_PAR].n_attr[(i%N_PAR)*NODE_DIM_HOST + dim] = (DTYPE)in[i * NODE_DIM_HOST + dim];
                    
                }
            }
        }else{
            cout<<"load node error"<<in_node_data<<" \n";
        }
        fin_n.close();
        
        if(fin_e.is_open()){
            
			std::cout<<"load with "<<in_edge_data<<" \n";
            std::vector<float> in;
            for (int r = 0; r < MAX_EDGE_HOST; r++){
                for (int c = 0; c < EDGE_DIM_HOST; c++){
                    float f;
                    fin_e >> f;  // Take input from file and put into myArray
                    in.push_back(f);
                }
            }
            
            
            for(int i=0; i<MAX_EDGE_HOST; i++){
                for(int dim=0; dim<EDGE_DIM_HOST; dim++){
                    ET_S[i/E_PAR].e_attr[(i%E_PAR)*EDGE_DIM_HOST + dim] = (DTYPE)in[i * EDGE_DIM_HOST + dim];
                }
            }
        }else{
            cout<<"load edge error"<<in_edge_data<<" \n";
        }
        fin_e.close();
        
        if(fin_i.is_open()){
            cout<<"start_C1"<<endl;
			
			std::cout<<"load with "<<in_idx_data<<" \n";
            
            std::vector<float> in;
            for (int r = 0; r < MAX_EDGE_HOST; r++){
                for (int c = 0; c < 2; c++){
                    float f;
                    fin_i >> f;  // Take input from file and put into myArray
                    cout<<r<<" "<<c<<" "<<f <<endl;
                    
                    in.push_back(f);
                }
            }
           
            for(int i=0; i<MAX_EDGE_HOST; i++){
                IT_S[i/E_PAR].i_attr[(i%E_PAR)*2 + 0] = (ap_uint<16>)in[i * 2 + 0];
                IT_S[i/E_PAR].i_attr[(i%E_PAR)*2 + 1] = (ap_uint<16>)in[i * 2 + 1];
            }
        }else{
            cout<<"load index error"<<in_idx_data<<" \n";
        }
        fin_i.close();
            
        /*
        if(fgold.is_open()){
            std::cout<<"load with "<<golden_data<<" \n";
            std::getline(fgold,iline);
            char* cstr=const_cast<char*>(iline.c_str());
            char* current;
            std::vector<int> in;
            current=strtok(cstr," ");
            while(current!=NULL) {
                in.push_back(atoi(current));
                current=strtok(NULL," ");
            }
            for(int i0 = 0; i0 < MAX_TOTAL_EDGE_HOST; i0++) { 
                golden_e_fpga[g][i0] = (bool)in[i0];
                //cout<<golden_e_fpga[g][i0] << " "<<in[i0]<<endl;
            } 
            
            //copy_data<int, bool, 0, MAX_TOTAL_EDGE_HOST*1>(in, golden_e_fpga[g]);
            fgold.close();
        }
        // Declare streams
        else{
            std::cout<<"ERROR: cannot load golden file\n";
        }
        */
        
    }
}


#endif
