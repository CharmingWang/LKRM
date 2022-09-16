import pickle
import argparse
import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', dest='path', type=str, default='vg_graph_r_a.pkl',
                    help='an integer for the accumulator')


args = parser.parse_args()



path=os.getcwd() + '/'+ args.path  #path='/root/……/aus_openface.pkl'   pkl文件所在路径
	   
f=open(path,'rb')
data=pickle.load(f)


#data[0][0] = 0

print(data)
print(len(data))

#pickle.dump(data, open( os.getcwd()  + '/'+ args.path + '_new', 'wb'))
