import os
from threading import main_thread
import numpy as np
import torch
def compute_avg_path_length(adj_list,length):
    print("length is {}".format(length))
    matrix = np.full([length,length],0)
    for i in range(length):
        matrix[i][i] = 1
        for j in range(i+1,length):
            if i in adj_list:
                for r in adj_list[i]:
                    if j in adj_list[i][r]:
                        matrix[i][j]=1
                        matrix[j][i]=1
    matrix = torch.from_numpy(matrix).float().cuda()
    re = torch.mm(matrix,matrix)
    print("1")
    re = torch.mm(re,matrix)
    print("2")
    re = torch.mm(re,matrix)
    print("3")
    re = torch.where(re > 0,1,0)
    re = re.cpu().numpy()
    print(re)
    reach = np.sum(re)
    print(reach-2)
    print((reach-2)/((length-2)**2))
    # for k in range(length):
    #     for i in range(length):
    #         for j in range(length):
    #             if matrix[i][j] > matrix[i][k]+matrix[k][j]:
    #                 matrix[i][j] = matrix[i][k]+matrix[k][j]
    # cnt = 0
    # path_length = 0
    # path_length_without1 = 0
    # for i in range(length):
    #     for j in range(i+1,length):
    #         if not np.isinf(matrix[i][j]):
    #             cnt += 1
    #             path_length+=matrix[i][j]
    #             if matrix[i][j] > 1:
    #                 path_length_without1 += matrix[i][j]
    # print("avg path length: {}".format(path_length/cnt))
    # print("avg path length without1: {}".format(path_length_without1/(cnt-(path_length-path_length_without1))))