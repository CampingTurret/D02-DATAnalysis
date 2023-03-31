from dataloader import data
import matplotlib.pyplot as plt

Pa = data('B Right',5,0.5,'cpu')
#Pb = data('B',5,0.5,'cuda')
#Pa.run_analysis_2D()
#Pa.Get_Dynamic()
#Pa.Split_Dynamic_Loaded(5)
#Pa.Plot_Raw_2D_Loaded("Time [s]","Bending [N-mm]")
#plt.show()
Pa.run_analysis_2D()
#Pb.run_analysis_2D_Quick('quick')


