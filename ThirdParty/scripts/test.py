import numpy as np
import cv2
import cupy as cp
import sys

def tester():
    result = np.array([[0, 1000, 1000, 2000, 2000]], dtype='float32')
    print(result.shape)
    print(result.dtype)
    print(result)
    return result
    

# import cv2
# import numpy as np
# import cupy as cp

# cpCudaMem = cp.cuda.memory
# source = op('source')
# topCudaMem = source.cudaMemory()

# numChans = 4 # rgba
# shape = (source.height, source.width, numChans)
# dType = cp.uint8

# offset = 0

# cpMemoryPtr = cpCudaMem.MemoryPointer(cpCudaMem.UnownedMemory(
# 				topCudaMem.ptr, topCudaMem.size, topCudaMem),
# 				offset)
				
# frameGPU = cp.ndarray(shape, dType, cpMemoryPtr)

# # just for fun, flip vertically on GPU
# frameGPU = frameGPU[::-1,: ,:] # flip vertical

# # copy GPU frame to CPU
# frameCPU =  cp.asnumpy(frameGPU)

# # write to Script TOP
# target = op('target')

# k = tester()