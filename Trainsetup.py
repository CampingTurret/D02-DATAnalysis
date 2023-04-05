from dataloader import data
from multiprocessing import Pool
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
    P = [['B Right',5,'Flap','cuda']]
    

    p = Pool(processes=(os.cpu_count() -2))
    with p as pool:
        pool.map(trainfunction, P)




