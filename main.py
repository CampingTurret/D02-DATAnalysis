from dataloader import data
import time


Pa = data('B',5,0.5)
#Pa.Split_static()
#print(Pa.Get_Static_Free())
Pa.run_analysis_2D()
