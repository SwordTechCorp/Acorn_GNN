#ifndef __UTIL_H__
#define __UTIL_H__
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <array>
#include <cmath>
#include "hls_math.h"
#include "su_defines.h"
#include "ap_int.h"
#include "ap_fixed.h"


using std::cout;
using std::endl;

#define WEIGHTS_DIR "/home/suyuru/myMP/mymy_v23_newalgo_mm/model_txt/"

template<class T, size_t SIZE_I, size_t SIZE_J, size_t SIZE_K, size_t SIZE_L>
void load_weights_from_txt_4d(T w[SIZE_I][SIZE_J][SIZE_K][SIZE_L], const char* fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    for(int m=0; m< SIZE_J * SIZE_L; m++){
        for(int n=0; n< SIZE_I * SIZE_K; n++){
            if( std::getline(infile, line)){
                double d;
                std::istringstream(line) >> d;
                w[m/SIZE_K][n/SIZE_L][m%SIZE_K][n%SIZE_L] = d;
            }
            else{
                cout<<"not enough line"<<m<<" "<<n<<endl;
            }
            
        }
    }
    
}
template<class T, size_t SIZE_I>
void load_weights_from_txt_1d(T w[SIZE_I], const char* fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    for(int m=0; m< SIZE_I; m++){
        std::getline(infile, line);
        double d;
        std::istringstream(line)>> d;
        w[m] = d;
    }
   
}
template <typename DTYPE>
void print_fixed(DTYPE e, int n_float){
    
    cout<<"print_fixed----------------"<<endl;
    for(auto item : e ){ 
        int item_int = item *  int(pow(2, n_float));
        float item_f = item_int / pow(2, n_float);   
        cout<<item_f<<" ";
    }
    cout<<endl;
}
constexpr int ceildiv_even(int dividend, int divisor)
{
//#pragma HLS INLINE
	int out = ((dividend + divisor - 1) / divisor);
	if(out% 2 ==1)
		return out + 1;
    else return out;
}
constexpr int ceildiv(int dividend, int divisor)
{
//#pragma HLS INLINE
    return (dividend + divisor - 1) / divisor;
}
template<class data_T,class weight_T,class res_T>
res_T product_dsp(
  data_T data,
  weight_T weight
){
  #pragma HLS INLINE
  res_T product;
  //#pragma HLS RESOURCE variable=product core=DSP48
  #pragma HLS BIND_OP variable=product op=mul impl=DSP latency=3
  product=data*weight;
  return product;
    
}
template<class data_T, class res_T,size_t n_in>
void  layer_norm(
    std::array<data_T, n_in> &data,
    std::array<res_T, n_in> &res
)
{
//#pragma HLS PIPELINE
#pragma HLS INLINE OFF
  data_T datareg;
  
  data_T  sum = 0;
  data_T  mean;
  
  data_T  squ_sum = 0;
  data_T  squ_tmp;
  data_T  std;
  
  
  
  for(int ii=0; ii<n_in; ii++) {
#pragma HLS UNROLL
    sum += data[ii];
    squ_sum += data[ii] * data[ii];
  }
  mean = sum / n_in;
  
  squ_tmp = squ_sum / n_in - mean * mean;
  std = hls::sqrt(squ_tmp);
  //fxp_sqrt(std, squ_tmp);

  for(int ii=0; ii<n_in; ii++) {
#pragma HLS UNROLL
    if(squ_tmp!=0)
        res[ii] = (data[ii] - mean) / squ_tmp;
    else 
        res[ii] = 0;
  }
  
}


template<class data_T, class res_T,size_t n_in>
void  relu(
    std::array<data_T, n_in> &data,
    std::array<res_T, n_in> &res
)
{
  //#pragma HLS PIPELINE
#pragma HLS INLINE OFF
  data_T datareg;
  for (int ii=0; ii<n_in; ii++) {
#pragma HLS UNROLL
      datareg = data[ii];
      if (datareg > 0) res[ii] = datareg;
      else res[ii] = 0;
  }
}

template <typename DTYPE>
DTYPE uint16_to_DTYPE(ap_uint<16> u) {
#pragma HLS inline
//#pragma HLS pipeline II=1
	DTYPE * tmpPointer_v = (DTYPE*) & u;
	return (*tmpPointer_v);
}

template <typename DTYPE>
ap_uint<16> DTYPE_to_uint16(DTYPE u) {
#pragma HLS inline
//#pragma HLS pipeline II=1
	ap_uint<16> * tmpPointer_v = (ap_uint<16>*) & u;
	return (*tmpPointer_v);
}

template <typename DTYPE>
DTYPE uint32_to_DTYPE(ap_uint<32> u) {
#pragma HLS inline
//#pragma HLS pipeline II=1
	DTYPE * tmpPointer_v = (DTYPE*) & u;
	return (*tmpPointer_v);
}

template <typename DTYPE>
ap_uint<32> DTYPE_to_uint32(DTYPE u) {
#pragma HLS inline
//#pragma HLS pipeline II=1
	ap_uint<32> * tmpPointer_v = (ap_uint<32>*) & u;
	return (*tmpPointer_v);
}

template <typename DTYPE>
float DTYPE_to_float(DTYPE u) {
#pragma HLS inline
//#pragma HLS pipeline II=1
	float * tmpPointer_v = (float*) & u;
	return (*tmpPointer_v);
}


//=============================================================//
// layer norm with LUT need check!! Error for RTL-synth!!      //
// sqrt_rec_float                                              //
// init_sqrt_rec_table                                         //
// layer_norm_LUT                                              //
//=============================================================//

inline float sqrt_rec_float(float input) {
    return 1./ hls::sqrt(input);
}

template<class table_t, int N_TABLE>
void init_sqrt_rec_table(table_t table_out[N_TABLE])
{
    //cout<<"sig_table_ini:============="<<endl;
    for (int ii = 0; ii < N_TABLE; ii++) {
        float in_val = (128.) *(ii)/float(N_TABLE);
        table_t real_val = sqrt_rec_float(in_val);
        table_out[ii] = real_val;
        
        //int val_int = real_val << 14;
        //float val_f = val_int / (16384.);
        
        //cout<<"idx: "<<ii<<" in_val: "<<in_val << "val: "<< real_val <<endl;
    }
    //cout<< "end table ini==========="<<endl;
}

template<class ATTR_T, size_t DIM, class table_t, int table_size>
table_t sqrt_rec_LUT(
    ATTR_T data
){
        // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    table_t sqrt_table[table_size];
#else
    static bool initialized = false;
    static table_t sqrt_table[table_size];
#endif
    if (!initialized) {
        init_sqrt_rec_table<table_t, table_size>(sqrt_table);
        initialized = true;
    }

#pragma HLS PIPELINE
    
    table_t data_round;
    int index;
    
    
    data_round = data * table_size / ATTR_T(128.);
    index = int(data_round);
    if (index < 0)   
       index = 0;
    else if (index > table_size-1) 
        index = table_size-1;
    
    //cout<< "in_val: "<<data<<endl;
    //cout<< "data_round: "<< data_round<<endl;
    //cout<<"idx: "<<index<<endl;
    
    return sqrt_table[index];
}
template<class data_T, class res_T, class table_t,size_t n_in, int table_size>
void  layer_norm_LUT(
    std::array<data_T, n_in> &data,
    std::array<res_T, n_in> &res
)
{
//#pragma HLS PIPELINE
#pragma HLS INLINE OFF

#ifdef __HLS_SYN__
    bool initialized = false;
    table_t sqrt_table[table_size];
#else
    static bool initialized = false;
    static table_t sqrt_table[table_size];
#endif
    if (!initialized) {
        init_sqrt_rec_table<table_t, table_size>(sqrt_table);
        initialized = true;
    }

  data_T datareg;
  
  ap_fixed<16,8>  sum = 0;
  
  ap_fixed<16,8>  squ_sum = 0;
  
  ap_fixed<32,16> k;
  ap_fixed<32,16> sqrt_rec_k;
  
  //cout<<"AAA"<<endl;
  
  
  for(int ii=0; ii<n_in; ii++) {
#pragma HLS UNROLL
    sum += data[ii];
    squ_sum += data[ii] * data[ii];
  }
  
  //cout<<"BBB "<<sum<<" "<< squ_sum<<endl;

  k = ap_fixed<32,16> (48 * squ_sum - sum*sum);
  //cout<< "in_val: "<<k<<endl;
  
  table_t data_round;
  int index;


  data_round = table_t (k * data_T(1024 / 128));
  //cout<< "data_round: "<< data_round<<endl;
  
  index = int(data_round);
  //cout<<"idx: "<<index<<endl; 
  
  if (index < 0)   
     index = 0;
  else if (index > 1024-1) 
      index = 1024-1;


  
  
  
  
  sqrt_rec_k = sqrt_table[index];
  
  //fxp_sqrt(std, squ_tmp);

  for(int ii=0; ii<n_in; ii++) {
#pragma HLS UNROLL
    res[ii] = (n_in * data[ii] - sum) * sqrt_rec_k;
  }
  
   /* cout<<"layer norm"<<endl;
  print_fixed(data, 4);
  cout<<sum<<endl;
  cout<<squ_sum<<endl;
  cout<<k<<endl;
  cout<<sqrt_rec_k<<endl;
  print_fixed(res, 4);
  cout<<"===========\n"; */
}  

  


#endif
