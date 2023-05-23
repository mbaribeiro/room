import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class Graphs:
    def __init__(self):
        self.coord = None
        self.predInputs = None
        self.predOutputs_Temp = None
        self.predOutputs_Vel = None
        self.cilinder = None
    
    def results(self):
        # Save predicted outputs
        for i in range(len(self.predInputs)):
            inTemp, inVel = self.predInputs[i]
            inTemp, inVel = round(inTemp), round(inVel)
            with open(f'./data/predicteds/T{inTemp}V{inVel}.csv', 'w') as out_file:
                out_file.write('i,j,k,temp,vel,count\n')
                predicteds = '\n'.join([f"{a},{b},{c}" for a, b, c in zip(self.coord, self.predOutputs_Temp[i], self.predOutputs_Vel[i])]).replace('[', '').replace(']', '')
                out_file.write(predicteds)

                df = pd.read_csv(f'./data/testOutputs/T{inTemp}V{inVel}.csv', delimiter='\t')

                # remove the lines in the output file that have all zeros in both outputs
                df = df[~np.all(df.iloc[:, 3:5] == 0, axis=1)]

                # Plot the predicted outputs and the real outputs with scatter in 3d with temperature and velocity
                fig1 = plt.figure()

                # Criate a boolean mask to remove the points with temperature less than a number
                mask = np.where(self.coord[:, 2] < 15)

                # First grath (Predicted Temp)
                ax1 = fig1.add_subplot(221, projection='3d')
                surf = ax1.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask],self.coord[:, 1][mask], c=self.predOutputs_Temp[i][mask], cmap='jet')
                ax1.set_xlabel('X Label')
                ax1.set_ylabel('Y Label')
                ax1.set_zlabel('Z Label')
                ax1.set_title('Predicted Temp')

                # Second grath (Real Temp)
                ax2 = fig1.add_subplot(222, projection='3d')
                c_array = np.array(df.iloc[:, 3])
                surf = ax2.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=c_array[mask], cmap='jet')
                ax2.set_xlabel('X Label')
                ax2.set_ylabel('Y Label')
                ax2.set_zlabel('Z Label')
                plt.colorbar(surf, ax=[ax1, ax2])
                ax2.set_title('Real Temp')

                # Third grath (Predicted Vel)
                ax3 = fig1.add_subplot(223, projection='3d')
                surf = ax3.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask],self.coord[:, 1][mask], c=self.predOutputs_Vel[i][mask], cmap='jet')
                ax3.set_xlabel('X Label')
                ax3.set_ylabel('Y Label')
                ax3.set_zlabel('Z Label')
                ax3.set_title('Predicted Vel')

                # Fourth grath (Real Vel)
                ax4 = fig1.add_subplot(224, projection='3d')
                c_array = np.array(df.iloc[:, 4])
                surf = ax4.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=c_array[mask], cmap='jet')
                ax4.set_xlabel('X Label')
                ax4.set_ylabel('Y Label')
                ax4.set_zlabel('Z Label')
                plt.colorbar(surf, ax=[ax3, ax4])
                ax4.set_title('Real Vel')

                # Plot the error between the predicted outputs and the real outputs
                fig2 = plt.figure()

                ax5 = fig2.add_subplot(121, projection='3d')
                c_array = np.array(df.iloc[:, 3])
                c = self.predOutputs_Temp[i][mask]
                diff1 = abs(c_array[mask] - c)
                surf = ax5.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=diff1, cmap='jet')
                ax5.set_xlabel('X Label')
                ax5.set_ylabel('Y Label')
                ax5.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax5)
                ax5.set_title('Error Temp')

                ax6 = fig2.add_subplot(122,  projection='3d')
                c_array = np.array(df.iloc[:, 4])
                c = self.predOutputs_Vel[i][mask]
                diff2 = abs(c_array[mask] - c)
                surf = ax6.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=diff2, cmap='jet')
                ax6.set_xlabel('X Label')
                ax6.set_ylabel('Y Label')
                ax6.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax6)
                ax6.set_title('Error Vel')

                # Plot only the 5% of the points with the highest error
                fig3 = plt.figure()

                ax7 = fig3.add_subplot(121, projection='3d')
                c_array = np.array(df.iloc[:, 3])
                c = self.predOutputs_Temp[i][mask]
                diff1 = abs(c_array[mask] - c)
                diff1 = np.array(diff1)
                diff1 = diff1.flatten()
                diff1.sort()
                diff1 = diff1[::-1]
                diff1 = diff1[:int(len(diff1)*0.05)]
                mask2 = np.isin(abs(c_array[mask] - c), diff1)
                surf = ax7.scatter(self.coord[:, 0][mask][mask2], self.coord[:, 2][mask][mask2], self.coord[:, 1][mask][mask2], c=diff1, cmap='jet')
                ax7.set_xlabel('X Label')
                ax7.set_ylabel('Y Label')
                ax7.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax7)
                ax7.set_title('Error Temp')

                ax8 = fig3.add_subplot(122,  projection='3d')
                c_array = np.array(df.iloc[:, 4])
                c = self.predOutputs_Vel[i][mask]
                diff2 = abs(c_array[mask] - c)
                diff2 = np.array(diff2)
                diff2 = diff2.flatten()
                diff2.sort()
                diff2 = diff2[::-1]
                diff2 = diff2[:int(len(diff2)*0.05)]
                mask2 = np.isin(abs(c_array[mask] - c), diff2)
                surf = ax8.scatter(self.coord[:, 0][mask][mask2], self.coord[:, 2][mask][mask2], self.coord[:, 1][mask][mask2], c=diff2, cmap='jet')
                ax8.set_xlabel('X Label')
                ax8.set_ylabel('Y Label')
                ax8.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax8)
                ax8.set_title('Error Vel')

                # Plot the histogram of the error
                fig4 = plt.figure()

                ax9 = fig4.add_subplot(121)
                ax9.hist(abs((np.array(df.iloc[:, 3]))[mask] - self.predOutputs_Temp[i][mask]), bins=100)
                ax9.set_title('Error Temp')

                ax10 = fig4.add_subplot(122)
                ax10.hist(abs((np.array(df.iloc[:, 4]))[mask] - self.predOutputs_Vel[i][mask]), bins=100)
                ax10.set_title('Error Vel')

                plt.show()

    def r2(self):
        # Plot the predicted outputs and the real outputs
        for i in range(len(self.predInputs)):
            inTemp, inVel = self.predInputs[i]
            inTemp, inVel = round(inTemp), round(inVel)
            output_file = f'./data/testOutputs/T{inTemp}V{inVel}.csv'
            df = pd.read_csv(output_file, delimiter='\t')

            # Temperature
            fig, (ax1, ax2) = plt.subplots(1, 2)
            output_list = df.iloc[:, 3].values.tolist()

            nonzero_indices = [i for i in range(len(output_list)) if output_list[i] != 0]
            output_list_filt = [output_list[i] for i in nonzero_indices]

            m, b = np.polyfit(output_list_filt, self.predOutputs_Temp[i], 1)
            x = np.array(output_list_filt)
            ax1.scatter(output_list_filt, self.predOutputs_Temp[i])
            ax1.plot(x, m*x + b, color='red')

            result = r2_score(output_list_filt, self.predOutputs_Temp[i])
            ax1.text(0.1, 0.9, "r-squared = {:.3f}".format(result), transform=ax1.transAxes)

            print("R² (Temperature) = " + str(result))
            ax1.set_title('Temperature [K]')
            ax1.set_ylabel('Predicted [K]')
            ax1.set_xlabel('Real [K]')

            # calculate the mse
            mse = tf.keras.losses.mean_squared_error(
                output_list_filt, self.predOutputs_Temp[i])
            print("MSE (Temperature) = " + str(mse))

            # Velocity
            output_list = df.iloc[:, 4].values.tolist()

            nonzero_indices = [i for i in range(len(output_list)) if output_list[i] != 0]
            output_list_filt = [output_list[i] for i in nonzero_indices]

            m, b = np.polyfit(output_list_filt, self.predOutputs_Vel[i], 1)
            x = np.array(output_list_filt)
            ax2.scatter(output_list_filt, self.predOutputs_Vel[i])
            ax2.plot(x, m*x + b, color='red')

            result = r2_score(output_list_filt, self.predOutputs_Vel[i])
            ax2.text(0.1, 0.9, "r-squared = {:.3f}".format(result), transform=ax2.transAxes)

            print("R² (Velocity) = " + str(result))
            ax2.set_title('Velocity [m s^-1]')
            ax2.set_ylabel('Predicted [m s^-1]')
            ax2.set_xlabel('Real [m s^-1]')

            # calculate the mse
            mse = tf.keras.losses.mean_squared_error(output_list_filt, self.predOutputs_Vel[i])
            print("MSE (Velocity) = " + str(mse))

            plt.show()

    def cilinderGraph(self):
        #plot the bars graph on a specific cilinder coordinate
        for i in range(len(self.predInputs)):
            inTemp, inVel = self.predInputs[i]
            inTemp, inVel = round(inTemp), round(inVel)
            with open(f'./data/predicteds/T{inTemp}V{inVel}.csv', 'w') as out_file:
                out_file.write('i,j,k,temp,vel,count\n')
                predicteds = '\n'.join([f"{a},{b},{c}" for a, b, c in zip(self.coord, self.predOutputs_Temp[i], self.predOutputs_Vel[i])]).replace('[', '').replace(']', '')
                out_file.write(predicteds)

                df = pd.read_csv(f'./data/testOutputs/T{inTemp}V{inVel}.csv', delimiter='\t')

                # remove the lines in the output file that have all zeros in both outputs
                df = df[~np.all(df.iloc[:, 3:5] == 0, axis=1)]

                # Plot the predicted outputs and the real outputs with scatter in 3d with temperature and velocity
                fig1 = plt.figure()

                # Criate a boolean mask to remove the outside a cilinder
                #self.cilinder = np.array([x, y, z, r, h, b(1=positive,-1=negative)])
                mask = np.where(((self.coord[:, 0] - self.cilinder[0])**2 + (self.coord[:, 2] - self.cilinder[2])**2 < self.cilinder[3]**2) & (np.where((self.cilinder[5] == 1) & (self.coord[:, 1] > self.cilinder[4]), True, self.coord[:, 1] < self.cilinder[4])))

                # First grath (Predicted Temp)
                ax1 = fig1.add_subplot(221, projection='3d')
                surf = ax1.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask],self.coord[:, 1][mask], c=self.predOutputs_Temp[i][mask], cmap='jet')
                ax1.set_xlabel('X Label')
                ax1.set_ylabel('Y Label')
                ax1.set_zlabel('Z Label')
                ax1.set_title('Predicted Temp')

                # Second grath (Real Temp)
                ax2 = fig1.add_subplot(222, projection='3d')
                c_array = np.array(df.iloc[:, 3])
                surf = ax2.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=c_array[mask], cmap='jet')
                ax2.set_xlabel('X Label')
                ax2.set_ylabel('Y Label')
                ax2.set_zlabel('Z Label')
                plt.colorbar(surf, ax=[ax1, ax2])
                ax2.set_title('Real Temp')

                # Third grath (Predicted Vel)
                ax3 = fig1.add_subplot(223, projection='3d')
                surf = ax3.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask],self.coord[:, 1][mask], c=self.predOutputs_Vel[i][mask], cmap='jet')
                ax3.set_xlabel('X Label')
                ax3.set_ylabel('Y Label')
                ax3.set_zlabel('Z Label')
                ax3.set_title('Predicted Vel')

                # Fourth grath (Real Vel)
                ax4 = fig1.add_subplot(224, projection='3d')
                c_array = np.array(df.iloc[:, 4])
                surf = ax4.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=c_array[mask], cmap='jet')
                ax4.set_xlabel('X Label')
                ax4.set_ylabel('Y Label')
                ax4.set_zlabel('Z Label')
                plt.colorbar(surf, ax=[ax3, ax4])
                ax4.set_title('Real Vel')

                # Plot the error between the predicted outputs and the real outputs
                fig2 = plt.figure()

                ax5 = fig2.add_subplot(121, projection='3d')
                c_array = np.array(df.iloc[:, 3])
                c = self.predOutputs_Temp[i][mask]
                diff1 = abs(c_array[mask] - c)
                surf = ax5.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=diff1, cmap='jet')
                ax5.set_xlabel('X Label')
                ax5.set_ylabel('Y Label')
                ax5.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax5)
                ax5.set_title('Error Temp')

                ax6 = fig2.add_subplot(122,  projection='3d')
                c_array = np.array(df.iloc[:, 4])
                c = self.predOutputs_Vel[i][mask]
                diff2 = abs(c_array[mask] - c)
                surf = ax6.scatter(self.coord[:, 0][mask], self.coord[:, 2][mask], self.coord[:, 1][mask], c=diff2, cmap='jet')
                ax6.set_xlabel('X Label')
                ax6.set_ylabel('Y Label')
                ax6.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax6)
                ax6.set_title('Error Vel')

                # Plot only the 5% of the points with the highest error
                fig3 = plt.figure()

                ax7 = fig3.add_subplot(121, projection='3d')
                c_array = np.array(df.iloc[:, 3])
                c = self.predOutputs_Temp[i][mask]
                diff1 = abs(c_array[mask] - c)
                diff1 = np.array(diff1)
                diff1 = diff1.flatten()
                diff1.sort()
                diff1 = diff1[::-1]
                diff1 = diff1[:int(len(diff1)*0.05)]
                mask2 = np.isin(abs(c_array[mask] - c), diff1)
                surf = ax7.scatter(self.coord[:, 0][mask][mask2], self.coord[:, 2][mask][mask2], self.coord[:, 1][mask][mask2], c=diff1, cmap='jet')
                ax7.set_xlabel('X Label')
                ax7.set_ylabel('Y Label')
                ax7.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax7)
                ax7.set_title('Error Temp')

                ax8 = fig3.add_subplot(122,  projection='3d')
                c_array = np.array(df.iloc[:, 4])
                c = self.predOutputs_Vel[i][mask]
                diff2 = abs(c_array[mask] - c)
                diff2 = np.array(diff2)
                diff2 = diff2.flatten()
                diff2.sort()
                diff2 = diff2[::-1]
                diff2 = diff2[:int(len(diff2)*0.05)]
                mask2 = np.isin(abs(c_array[mask] - c), diff2)
                surf = ax8.scatter(self.coord[:, 0][mask][mask2], self.coord[:, 2][mask][mask2], self.coord[:, 1][mask][mask2], c=diff2, cmap='jet')
                ax8.set_xlabel('X Label')
                ax8.set_ylabel('Y Label')
                ax8.set_zlabel('Z Label')
                plt.colorbar(surf, ax=ax8)
                ax8.set_title('Error Vel')

                # Plot the histogram of the error
                fig4 = plt.figure()

                ax9 = fig4.add_subplot(121)
                ax9.hist(abs((np.array(df.iloc[:, 3]))[mask] - self.predOutputs_Temp[i][mask]), bins=100)
                ax9.set_title('Error Temp')

                ax10 = fig4.add_subplot(122)
                ax10.hist(abs((np.array(df.iloc[:, 4]))[mask] - self.predOutputs_Vel[i][mask]), bins=100)
                ax10.set_title('Error Vel')

                plt.show()

