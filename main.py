from dataloader import data
import time


Pa = data('B',5,0.5)
#Pa.Split_static()
#print(Pa.Get_Static_Free())

fileselect = 5


Pa.Get_Dynamic()
Pa.Split_Dynamic_Loaded(fileselect)
print(Pa.dynamicloaded[fileselect,1])
Pa.Train_Dynamic_models_2D_Loaded(["Time [s]"],["Pot [degree]","Bending [N-mm]"],1000,0.01)
#print(Pa.dynamicloaded[fileselect,1])
#time.sleep(3)
#print(Pa.dynamicsplit)