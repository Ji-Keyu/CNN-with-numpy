from .layers import *
from cs231n.fast_layers import *
from nndl.conv_layers import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def affine_relu_forward(x, w, b):
  """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db




def conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  out, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, bn_cache = spatial_batchnorm_forward(out, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out)
  out, pool_cache = max_pool_forward_fast(out, pool_param)

  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  dout = max_pool_backward_fast(dout, pool_cache)
  dout = relu_backward(dout, relu_cache)
  dout, dgamma, dbeta = spatial_batchnorm_backward(dout, bn_cache)
  dx, dw, db = conv_backward_fast(dout, conv_cache)

  return dx, dw, db, dgamma, dbeta