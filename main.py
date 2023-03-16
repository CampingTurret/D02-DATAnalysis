from dataloader import data
import time


Pa = data('C',5,8)
#Pa.Split_static()
#print(Pa.Get_Static_Free())
Pa.run_analysis_2D()
