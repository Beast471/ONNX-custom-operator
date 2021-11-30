#include <fstream>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>

/// Implements the repeat interleave operation
/// param input: torch 1-dimensional Tensor
/// return torch::Tensor: interleaved 1-dimensional result
torch::Tensor repeatInterleave(torch::Tensor input_tensor) 
{
  int size_desired = 512;
  int size_present = input_tensor.sizes()[1];
  torch::Tensor output_array = torch::zeros({1, 512});
  int interleave_factor = size_desired/size_present;
  int main_iter=512;
  
//If the input tensor is already of len 512, then return  
  if(size_present == 512)
  return input_tensor.clone();

  int inner_iter=0;
  int size_present_iter=0;//This iterator goes from to sizeof(input_tensor) -> increments by 1 for every while loop
  int temp=0;

//This loop runs for : 512 divided by interleave_factor  
  while(main_iter>0)
  {
    for(inner_iter=0;inner_iter<interleave_factor;inner_iter++){
      temp=512-main_iter;
      output_array[0][temp+inner_iter]=input_tensor[0][size_present_iter];
    }
  
    size_present_iter++;
    main_iter = main_iter-interleave_factor;
  }
 
  return output_array;
}



/// Implements the Reduction ONNX operator by interleaving the
/// param layerOne: layer 1 output after AdaptiveAvgPooling [1x64] or [64]
/// param layerTwo: layer 2 output after AdaptiveAvgPooling [1x128] or [128]
/// param layerThree: layer 3 output after AdaptiveAvgPooling [1x256] or [256]
/// param layerFour: layer 4 output after AdaptiveAvgPooling [1x512] or [512]
/// return torch::Tensor: reduction output of the tensors [1x512] or [512]
torch::Tensor reduction (
 torch::Tensor layerOne,
 torch::Tensor layerTwo,
 torch::Tensor layerThree,
 torch::Tensor layerFour
 
) {
torch::Tensor reduced_tensor;

//Run interleave for first 3 vectors([1,64],[1,128] and [1,256]) because the 4th vector is of size [1,512]
layerOne = repeatInterleave(layerOne);
layerTwo = repeatInterleave(layerTwo);
layerThree = repeatInterleave(layerThree);

//cat operation creates [4,512] vector from the input, then mean creates a [512]
reduced_tensor = torch::cat({layerOne,layerTwo, layerThree, layerFour});
reduced_tensor = torch::mean(reduced_tensor,0);

return reduced_tensor;
}

//registering the namespace and operator name that will be available in python
static auto registry = torch::RegisterOperators("mynamespace::reduction", &reduction);
