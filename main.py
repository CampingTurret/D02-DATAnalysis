from dataloader import data, thirddimdata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

Pa = data('A',0,0.5,'cuda')
Pa.run_analysis_2D_Quick(show=False)
plt.clf()

Pa.Plot_Raw_2D_All(['Locked'])
print(len(Pa.dynamicsplit[1]))
Pa.run_analysis_2D_Quick(mode ='limit',types = ['Locked'],show=False)
plt.xlabel('Time [s]')
plt.ylabel('Bending moment [Nmm]')
plt.legend()
plt.savefig(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'.','A005lockedraw.svg')))
plt.show()

#Pb = data('B Right',5,0.5,'cuda')
#A = Pb.Get_Dynamic()
#for i in A:
#    print(np.max(i[0]['Bending [N-mm]']))
#    print(i[1])

#Pa.run_analysis_2D()
#Pa.Get_Dynamic()
#Pa.Split_Dynamic_Loaded(5)
#Pa.Plot_Raw_2D_Loaded("Time [s]","Bending [N-mm]")
#plt.show()


