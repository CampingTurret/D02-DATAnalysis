from dataloader import data
from multiprocessing import Pool
import multiprocessing as mp
import time
import os


def trainfunction(l:list):
    Plate = l[0]
    AOA = l[1]
    hz = l[2]
    device = l[3]
    Pa = data(Plate,AOA,hz,device)
    Pa.run_Train_2D()
    return

if __name__ == '__main__':
    #Change the DATA's !!!
    P = [['B Right',0,0.5,'cuda'],['B Right',0,5,'cpu'],['B Right',0,8,'cpu']]
    

    p = Pool(processes=(os.cpu_count() -2))
    with p as pool:
        pool.map(trainfunction, P)




