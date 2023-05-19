class saveModels:
    def __init__(self):
        self.modelTemp = None
        self.modelVel = None

    def saveModels(self):
        # Save the models
        self.modelTemp.save('room/modelsResults/modelTemp.h5')
        self.modelVel.save('room/modelsResults/modelVel.h5')

