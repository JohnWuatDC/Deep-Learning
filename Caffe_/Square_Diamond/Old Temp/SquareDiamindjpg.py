############################################CREATE TRANING###############################################################import os
import os
import caffe
import types
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
############################################Pre Processing Data##############################################################
my_root = '/home/ajafari/Desktop/Caffe_Files/caffe_example/SquareDiamondjpg'
os.chdir(my_root)
############################################Pre Processing Data##############################################################

fo = open('diamond.txt', 'r')

file_content = fo.read().strip()
file_content = file_content.replace('\r\n', ';')
file_content = file_content.replace('\n', ';')
file_content = file_content.replace('\r', ';')
diamond = numpy.matrix(file_content)

fo.close()

fo = open('square.txt', 'r')

file_content = fo.read().strip()
file_content = file_content.replace('\r\n', ';')
file_content = file_content.replace('\n', ';')
file_content = file_content.replace('\r', ';')
square = numpy.matrix(file_content)

fo.close()

#scipy.misc.imsave('square.jpg', square)
#scipy.misc.imsave('diamond.jpg', diamond)
############################################Pre Processing Data##############################################################

#----------------------------Becareful about the Directories------------------------------------
#----------If you change Directory Change the Directories in a Shell Files too------------------

#os.system("sh create_imagenet.sh")
#os.system("sh make_imagenet_mean.sh")

#######################################Train the Network with the Solver######################################################

caffe.set_device(0)
caffe.set_mode_gpu()

solver =  caffe.SGDSolver('Square_diamond_solver1.prototxt')
#----------You need to run the following command to goustat works-------------------------------
#  sudo pip install gpustat
#os.system("gpustat")
#solver.solve()
#solver.net.forward()  # train net
#solver.test_nets[0].forward()  # test net (there can be more than one)

#solver.net.backward()
os.system("gpustat")
#solver.step(1)
net = solver.net

#solver.solve()
#----------You need to run the following command to goustat works-------------------------------
#  sudo pip install gpustat
#os.system("gpustat")
#----------------------------------------------------------------------------------------------
#---------------------------------------Training Caffe-----------------------------------------
solver.solve()
solver.net.forward()  # train net
#solver.test_nets[0].forward()  # test net (there can be more than one)
solver.net.backward()
#solver.step(1)
#solver = caffe.SGDSolver('solver1.prototxt')


#solver = caffe.SGDSolver('solver1.prototxt')