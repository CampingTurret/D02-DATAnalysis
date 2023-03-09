from dataloader import data
import time


Pa = data('A')
#Pa.Split_static()
#print(Pa.Get_Static_Free())
AOA = 5
Frq = 0.5
fileselect = 3


Pa.Get_Dynamic(AOA,Frq)
Pa.Split_Dynamic_Loaded(fileselect,Frq)
Pa.Train_Dynamic_models_2D_Loaded(["Time [s]"],["Pot [degree]","Bending [N-mm]"])
#print(Pa.dynamicloaded[fileselect,1])
#time.sleep(3)
#print(Pa.dynamicsplit)