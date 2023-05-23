import numpy as np
import pandas as pd

class ReadData:
    def __init__(self):
        self.pathInput = './data/inputs/inputs.csv'
        self.pathInputsPred = './data/inputs/inputsPred.csv'
        self.predInputs = None

    def readInput(self):  
        # Load input data
        with open(self.pathInput, 'r') as file:
            # Skip the header
            file.readline()
            # Read data lines
            data = []
            for line in file:
                values = [float(x) for x in line.strip().split(',')]
                data.append(values)
            # Convert data to numpy array
            data = np.array(data)
            print("reading inputs.csv...")
            return data
    
    def trainData(self):
        # Load input data
        with open('./data/inputs/inputs.csv', 'r') as f:
            # Skip the header
            next(f)
            data1 = []
            data2 = []
            for line in f:
                # Read input values
                inTemp, inVel = map(str, line.strip().split(','))

                # Load output data
                output_file = open(f'./data/outputs/T{inTemp}V{inVel}.csv', 'r')
                print("reading " + output_file.buffer.name + "...")
                df = pd.read_csv(output_file, delimiter='\t')
                coordenates = df.iloc[:, 0:3].values.tolist()
                data_line1 = df.iloc[:, 3].astype(str).tolist()
                data_line2 = df.iloc[:, 4].astype(str).tolist()
                data1 = data1 + [data_line1]
                data2 = data2 + [data_line2]

        return data1, data2, coordenates
    
    def readInputsPred(self):
        # Load input data
        with open(self.pathInputsPred, 'r') as file:
            # Skip the header
            file.readline()
            # Read data lines
            data = []
            for line in file:
                values = [float(x) for x in line.strip().split(',')]
                data.append(values)
            # Convert data to numpy array
            data = np.array(data)
        return data
    
    def readOutputsTemp(self):
        c_array = []
        for i in range(len(self.predInputs)):
            inTemp, inVel = self.predInputs[i]
            df = pd.read_csv(f'./data/testOutputs/T{round(inTemp)}V{round(inVel)}.csv', delimiter='\t')

            # remove the lines in the output file that have all zeros in both outputs
            df = df[~np.all(df.iloc[:, 3:5] == 0, axis=1)]
            
            c_array.append(df.iloc[:, 3].values.tolist())

        return c_array
    
    def readOutputsVel(self):
        c_array = []
        for i in range(len(self.predInputs)):
            inTemp, inVel = self.predInputs[i]
            df = pd.read_csv(f'./data/testOutputs/T{round(inTemp)}V{round(inVel)}.csv', delimiter='\t')

            # remove the lines in the output file that have all zeros in both outputs
            df = df[~np.all(df.iloc[:, 3:5] == 0, axis=1)]
            
            c_array.append(df.iloc[:, 4].values.tolist())

        return c_array