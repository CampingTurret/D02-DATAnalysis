from dataloader import data
import time


Pa = data('B Right',0,'Flap','cuda')
#Pb = data('B',5,0.5,'cuda')
#Pa.run_analysis_2D()
#Pa.Get_Dynamic()
#Pa.Split_Dynamic_Loaded(0)
#Pa.Plot_Raw_2D_Loaded("Time [s]","Bending [N-mm]")
Pa.run_analysis_2D()
#Pb.run_analysis_2D_Quick('quick')


