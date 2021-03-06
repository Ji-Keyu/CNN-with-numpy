import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride
  if not Hp.is_integer() or not Wp.is_integer():
    print("Invalid stride with given dimension and padding.")
    exit(1)
  else:
    Hp = int(Hp)
    Wp = int(Wp)
  out = np.zeros((N, F, Hp, Wp))
  
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)

  for i in np.arange(N):
    for j in np.arange(F):
      for k1 in np.arange(Hp):
        for k2 in np.arange(Wp):
          out[i,j,k1,k2] = np.sum(xpad[i,:,k1*stride:k1*stride+HH,k2*stride:k2*stride+WW]*w[j])+b[j]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dx = np.zeros(xpad.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)
  for i in np.arange(N):
    for j in np.arange(F):
      for k1 in np.arange(out_height):
        for k2 in np.arange(out_width):
          db[j] += dout[i,j,k1,k2]
          
          dx[i,:,k1*stride:k1*stride+f_height,k2*stride:k2*stride+f_width] += dout[i,j,k1,k2]*w[j]

          dw[j] += xpad[i,:,k1*stride:k1*stride+f_height,k2*stride:k2*stride+f_width]*dout[i,j,k1,k2]
          #notice that these two very similar with forward

  dx = dx[:,:,pad:-pad,pad:-pad]
  #"unpad"
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  Hp = 1 + (H - HH) / stride
  Wp = 1 + (W - WW) / stride
  if not Hp.is_integer() or not Wp.is_integer():
    print("Invalid stride with given dimension.")
    exit(1)
  else:
    Hp = int(Hp)
    Wp = int(Wp)

  out = np.zeros((N, C, Hp, Wp))
  
  for i in np.arange(N):
    for j in np.arange(C):
      for k1 in np.arange(Hp):
        for k2 in np.arange(Wp):
          out[i,j,k1,k2] = np.max(x[i,j,k1*stride:k1*stride+HH,k2*stride:k2*stride+WW])
          

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  Hp = 1 + (H - HH) / stride
  Wp = 1 + (W - WW) / stride
  if not Hp.is_integer() or not Wp.is_integer():
    print("Invalid stride with given dimension.")
    exit(1)
  else:
    Hp = int(Hp)
    Wp = int(Wp)

  dx = np.zeros((N, C, H, W))
  
  for i in np.arange(N):
    for j in np.arange(C):
      for k1 in np.arange(Hp):
        for k2 in np.arange(Wp):
          indexmax = np.argmax(x[i,j,k1*stride:k1*stride+HH,k2*stride:k2*stride+WW])
          dx[i,j,k1*stride+int(indexmax/WW), k2*stride + indexmax%WW] = dout[i,j,k1,k2]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  x = x.reshape((N,C,H*W))  #flatten lower 2 dims
  x = x.transpose((1,2,0))  #pull out filter dim
  x = x.reshape((C,N*H*W))  #flatten lower dims
  x = x.transpose()         #T to fit arg format
  out, cache = batchnorm_forward(x, gamma, beta, bn_param)
  out = out.transpose().reshape((C,H*W,N)).transpose((2,0,1)).reshape(N,C,H,W)
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout = dout.reshape((N,C,H*W))
  dout = dout.transpose((1,2,0))
  dout = dout.reshape((C,N*H*W))
  dout = dout.transpose()
  dx, dgamma, dbeta = batchnorm_backward(dout, cache)
  dx = dx.transpose().reshape((C,H*W,N)).transpose((2,0,1)).reshape(N,C,H,W)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta