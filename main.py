from dataloader import data
import time


Pa = data('B',0,0.5,'cuda')
Pb = data('B',5,0.5,'cuda')
#Pa.run_analysis_2D()
Pa.run_analysis_2D_Quick('quick',show = False)
Pb.run_analysis_2D_Quick('quick')


