from dataloader import data
import time


Pa = data('C',5,8)
Pa.Get_Dynamic()
Pa.Split_Dynamic_Loaded(2)
Pa.Remove_Gaps_Dynamic()
Pa.Plot_Raw_2D_Loaded("Time [s]","Bending [N-mm]")
#Pa.Split_static()
#print(Pa.Get_Static_Free())
#Pa.run_analysis_2D()
