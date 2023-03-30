from dataloader import data


Pa = data('B Right',0,5,'cuda')
#Pb = data('B',5,0.5,'cuda')
#Pa.run_analysis_2D()
#Pa.Get_Dynamic()
#Pa.Split_Dynamic_Loaded(0)
#Pa.Plot_Raw_2D_Loaded("Time [s]","Bending [N-mm]")
Pa.run_analysis_2D_Quick()
#Pb.run_analysis_2D_Quick('quick')


