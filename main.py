from dataloader import data, thirddimdata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Pa = thirddimdata('B Right',5,'cuda')
Pb = data('B Right',5,0.5,'cuda')
A = Pb.Get_Dynamic()
for i in A:
    print(np.max(i[0]['Bending [N-mm]']))
    print(i[1])

#Pa.run_analysis_2D()
#Pa.Get_Dynamic()
#Pa.Split_Dynamic_Loaded(5)
#Pa.Plot_Raw_2D_Loaded("Time [s]","Bending [N-mm]")
#plt.show()
#Pa.Run_Train_Solo('Free')
#Pa.Generate_Plot('Free')
#Pb.run_analysis_2D_Quick('quick')


