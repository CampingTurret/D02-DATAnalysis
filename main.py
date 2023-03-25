from dataloader import data
import time


Pa = data('B',5,0.5,'cuda')
#Pa.Get_Dynamic()
#Pa.Split_Dynamic_Loaded(0)
#Pa.Remove_Gaps_Dynamic()
#Pa.Plot_Raw_2D_Loaded("Time [s]","Bending [N-mm]")
#Pa.run_analysis_2D()
Pa.run_analysis_2D_Quick()


