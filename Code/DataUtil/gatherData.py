import os
import pandas as pd
import shutil
import json
import numpy as np

class DataGathere:
    def __init__(self, rawDataPath, fimeDataPath, datasets):
        self.rawDataPath = rawDataPath
        self.fimeDataPath = fimeDataPath
        self.datasets = datasets
        self.imageDataPath = None
        self.textData = None
        self.explanationData = None

    def getTextMEMEX(self, dataset):
        textDict = {}
        explanationDict = {}
        # expPath = os.path.join(self.rawDataPath, dataset, 'MemeExplanationData', 'annotations.csv')
        # edf = pd.read_csv(expPath,encoding='utf-8')
        # imageNames = edf['image'].to_list()
        # explanations = edf['evidence'].to_list()
        # expDict = {}
        # for imageName, explaination in zip(imageNames, explanations):
        #     expDict[imageName] = explaination
        for fileType in ['train', 'test']:    
            csvPath = os.path.join(self.rawDataPath, dataset, 'MemeExplanationData', fileType + '.csv')
            df = pd.read_csv(csvPath, encoding='utf-8')
            imageNames = df['image'].to_list()
            ocrTexts = df['ocr_text'].to_list()
            df['evidence'].fillna('NoExp', inplace=True)
            explanations = df['evidence'].to_list()
            sentences = df['sentences'].to_list()
            lables = df['labels'].to_list()
            for expi, exp in enumerate(explanations):
                if exp == 'NoExp':
                    explanations[expi] = "Bhiman"
            for imageName, ocrText, explanation in zip(imageNames, ocrTexts, explanations):
                textDict[imageName] = {"ocr_text": ocrText, "caption": "", "title": ""}
                explanationDict[imageName] = explanation     
        return textDict, explanationDict
    
    def getData(self):
        for dataset in datasets:
            if dataset == "MEMEX-MCC":
                dataset_dict = {}
                self.imageDataPath = os.path.join(self.rawDataPath, dataset, 'MemeExplanationData', 'images')
                self.textData, self.explanationData =  self.getTextMEMEX(dataset)
                cleanDatasetPath = os.path.join(self.fimeDataPath, dataset)
                if os.path.exists(cleanDatasetPath):
                    shutil.rmtree(cleanDatasetPath)
                os.mkdir(cleanDatasetPath)
                newImagedataPath = os.path.join(cleanDatasetPath, 'img')
                os.mkdir(newImagedataPath)
                for imageName in os.listdir(self.imageDataPath):
                    newImagePath = os.path.join(newImagedataPath, dataset + '_' + imageName)
                    oldImagePath = os.path.join(self.imageDataPath, imageName)
                    if imageName in self.textData:
                        dataset_dict[dataset + '_' + imageName] = {'imgPath':newImagePath, 'text': self.textData[imageName], 'explanation':self.explanationData[imageName]}
                        shutil.copy(oldImagePath, newImagePath)
                    else:
                        print(f'{imageName}: Text Data Not found')
                jsonDumpFile = os.path.join(cleanDatasetPath, dataset+'.json')
                with open(jsonDumpFile, "w", encoding="utf-8") as fp:
                    json.dump(dataset_dict , fp, indent=2, sort_keys=True) 

if __name__ == "__main__":
    rawDataPath = os.path.join('..', '..', 'Data', 'RawData')
    fimeDataPath = os.path.join('..', '..', 'Data', 'FimeData')
    datasets = ['MEMEX-MCC']
    dgObj = DataGathere(rawDataPath=rawDataPath, fimeDataPath=fimeDataPath, datasets=datasets)
    dgObj.getData()



