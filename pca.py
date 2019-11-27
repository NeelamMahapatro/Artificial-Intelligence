from matplotlib import pyplot as plt
import numpy as np
from os import listdir
import operator
#training
testing_tensor = np.ndarray(shape=(112*92, 280), dtype=np.integer)
j=0
for i in range(40):
	img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/1.pgm")
	testing_tensor[:, j] = np.array(img[:, :]).flatten(); j+=1
	img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/2.pgm")
	testing_tensor[:, j] = np.array(img[:, :]).flatten(); j+=1
	img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/3.pgm")
	testing_tensor[:, j] = np.array(img[:, :]).flatten(); j+=1
	img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/4.pgm")
	testing_tensor[:, j] = np.array(img[:, :]).flatten(); j+=1
	img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/5.pgm")
	testing_tensor[:, j] = np.array(img[:, :]).flatten(); j+=1
	img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/6.pgm")
	testing_tensor[:, j] = np.array(img[:, :]).flatten(); j+=1
	img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/7.pgm")
	testing_tensor[:, j] = np.array(img[:, :]).flatten(); j+=1

mean = testing_tensor.mean(0)
mean_shifted = np.ndarray(shape=(112*92, 280), dtype=np.integer)
for i in range(280):
	temp = np.ndarray(shape=(112*92), dtype=np.integer)
	temp.fill(mean[i])
	mean_shifted[:,i] = testing_tensor[:,i] - temp

testing_tensor = np.transpose(testing_tensor)
cov_mat = np.cov(testing_tensor)
eig_val, eig_vec = np.linalg.eig(cov_mat)
#pca_mul = eig_vec[:,np.where(np.amax(eig_val))].flatten()
#pca_mul = pca_mul[:-100]
map_eigen = dict()

for i in range(len(eig_val)):
   map_eigen[eig_val[i]] = eig_vec[i]

sorted_dic = dict(sorted(map_eigen.items(), key = operator.itemgetter(0), reverse = True))

x = np.zeros(280, dtype=np.int) 
# corresponding y axis values 
y = np.zeros(280, dtype=float) 
  
# plotting the points  

  

for j in range(1, 281): 
    x[j] = j
    p=j
    sorted_eigen = np.zeros((p, 280), dtype = float)
    
    k=0
    for i in sorted_dic:
        sorted_eigen[k] = sorted_dic[i]
        k=k+1
        if(k>=p):
            break
    print(sorted_eigen.shape)
    pca_res = np.dot(sorted_eigen,testing_tensor)
    #pca_res = np.dot(np.transpose(eig_vec),testing_tensor)
    sig = np.ndarray(shape=(p,p), dtype=np.integer)
    for i in range(p):
    	sig[:,i] = np.dot(pca_res,mean_shifted[:,i])
    
    #pca_res = pca_res.reshape(112,92)
    #plt.imshow(pca_res,cmap='gray')
    #plt.show()
    '''
    print("Input Image Matrix size", str(testing_tensor.shape))
    print("Mean shifted Matrix size", str(mean_shifted.shape))
    print("Eigen Vector (Feature Vector) shape: ", str(eig_vec.shape))
    print("PCA result shape: ", str(pca_res.shape))
    print("Signature shape: ", str(sig.shape))
    '''
    correct=0
    num_images = 40
    for i in range(num_images):    
        img = plt.imread("./dataset/orl_faces/s"+str(i+1)+"/9.pgm")[:,:].flatten()
        mn = img.mean(0)
        temp = np.ndarray(shape=(112*92), dtype=np.integer)
        temp.fill(mn)
        mean_s = img - temp
        sig_sample = np.dot(pca_res,mean_s)
        #print("Signature of test face", str(sig_sample.shape))
        ind = 0
        min_dis = np.sqrt(np.sum(np.square(sig[:,0]-sig_sample)))
        for j in range(p):
            dist = np.sqrt(np.sum(np.square(sig[:,j]-sig_sample)))
            if dist<min_dis:
                min_dis = dist
                ind = j
        face_index = int((ind/7))+1
        matched_index =(ind%7)+1
        #print("Target Face "+str(i+1)+" matched with face index "+str(face_index)+" with score "+str(min_dis))
        
        '''
        plt.subplot(121)
        plt.imshow(plt.imread("./dataset/orl_faces/s"+str(i+1)+"/9.pgm"),cmap='gray')
        plt.subplot(122)
        plt.imshow(plt.imread("./dataset/orl_faces/s"+str(face_index)+"/"+str(matched_index)+".pgm"),cmap='gray')
        plt.show()
        '''
        if i+1 == face_index:
            correct=correct+1
            
    accuracy = correct/num_images
    print("Accuracy is", str(accuracy*100))
    y[j] = accuracy*100
plt.plot(x, y) 
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('Accuracy graph!') 
  
# function to show the plot 
plt.show()
