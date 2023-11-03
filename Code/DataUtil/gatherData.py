import os
import pandas as pd
import shutil
import json
import numpy as np
import ast
import copy

class DataGathere:
    def __init__(self, rawDataPath, fimeDataPath, datasets):
        self.rawDataPath = rawDataPath
        self.fimeDataPath = fimeDataPath
        self.datasets = datasets
        self.imageDataPath = None
        self.textData = None
        self.explanationData = None
        self.cleanDatasetPath = None
        self.newImagedataPath = None
        self.unifiedFormat = {"ocr_text": "", "caption": "", "title": "", "entityRole": {"hero": [], "villain": [], "victim": [], "other": []}}

    def getTextMEMEX(self, dataset):
        textDict = {}
        explanationDict = {}
        for fileType in ['test']:    
            csvPath = os.path.join(self.rawDataPath, dataset, 'MemeExplanationData', fileType + '.csv')
            df = pd.read_csv(csvPath, encoding='utf-8')
            imageNames = df['image'].to_list()
            ocrTexts = df['ocr_text'].to_list()
            df['evidence'].fillna('NoExp', inplace=True)
            explanations = df['evidence'].to_list()
            sentences = df['sentences'].to_list() #candidate evidences/explanations
            labels = df['labels'].to_list() #1 for actual evidence/explanation
            for expi, (exp, sentence, label) in enumerate(zip(explanations, sentences, labels)):
                if exp == 'NoExp':
                    #original dataset has not extracted actual explanations from candidates, so we do that here
                    sentence = ast.literal_eval(sentence)
                    label = ast.literal_eval(label)
                    if len(sentence) == len(label):
                        explanations[expi] = " ".join([sentence[i] for i in label if i == 1])
                        pass
                    else:
                        print("Length Missmatch")
                        explanations[expi] = "NoExp"
            for imageName, ocrText, explanation in zip(imageNames, ocrTexts, explanations):
                if explanation != "NoExp":
                    textDict[imageName] = copy.deepcopy(self.unifiedFormat)
                    textDict[imageName]["ocr_text"] = ocrText
                    explanationDict[imageName] = explanation     
        return textDict, explanationDict
    
    def getTextHVV(self, dataset):
        textDict = {}
        explanationDict = {}
        for fileType in ["val.jsonl"]:
            df = pd.read_json(path_or_buf=os.path.join(self.rawDataPath, dataset, "annotations_HVV", fileType), lines=True, encoding='utf-8')
            # print(df.head())
            for index, row in df.iterrows():
                textDict[row["image"]] = copy.deepcopy(self.unifiedFormat)
                textDict[row["image"]]["ocr_text"] = row["OCR"]
                for role in ["hero", "villain", "victim", "other"]:
                    textDict[row["image"]]["entityRole"][role] = row[role]
                explanationDict[row["image"]] = ""
        return textDict, explanationDict

    def createCleanDataDir(self, dataset):
        #create directory to store cleaned data
        self.cleanDatasetPath = os.path.join(self.fimeDataPath, dataset)
        if os.path.exists(self.cleanDatasetPath):
            shutil.rmtree(self.cleanDatasetPath)
        os.mkdir(self.cleanDatasetPath)
        #image will always be stored in 'img' directory
        self.newImagedataPath = os.path.join(self.cleanDatasetPath, 'img')
        os.mkdir(self.newImagedataPath)
    
    def mergeData(self, dataset):
        dataset_dict = {}
        #collect those images whose texts/explanations are present
        for imageName in os.listdir(self.imageDataPath):
            newImagePath = os.path.join(self.newImagedataPath, dataset + '_' + imageName)
            oldImagePath = os.path.join(self.imageDataPath, imageName)
            if imageName in self.textData:
                dataset_dict[dataset + '_' + imageName] = {'imgPath':newImagePath, 'text': self.textData[imageName], 'explanation':self.explanationData[imageName]}
                shutil.copy(oldImagePath, newImagePath)
            else:
                print(f'{imageName}: Text Data Not found')
        
        #write the cleaned data
        jsonDumpFile = os.path.join(self.cleanDatasetPath, dataset+'.json')
        with open(jsonDumpFile, "w", encoding="utf-8") as fp:
            json.dump(dataset_dict , fp, indent=2, sort_keys=True)
    
    def getData(self):
        for dataset in datasets:
            if dataset == "MEMEX-MCC":
                print(f'Start gathering dataset {dataset}')

                self.imageDataPath = os.path.join(self.rawDataPath, dataset, 'MemeExplanationData', 'images')
                
                #get text data in desired format from the original dataset
                self.textData, self.explanationData =  self.getTextMEMEX(dataset)
                
                #create directory to store cleaned data
                self.createCleanDataDir(dataset)
                
                # merge all data and write
                self.mergeData(dataset)
                
                print(f'Gathering dataset {dataset} Finished!') 
            
            if "HVV-" in dataset:
                print(f'Start gathering dataset {dataset}')
                
                self.imageDataPath = os.path.join(self.rawDataPath, dataset, 'images')

                #get text data in desired format from the original dataset
                self.textData, self.explanationData = self.getTextHVV(dataset)

                #create directory to store cleaned data
                self.createCleanDataDir(dataset)

                # merge all data and write
                self.mergeData(dataset)

                print(f'Gathering dataset {dataset} Finished!')  

if __name__ == "__main__":
    rawDataPath = os.path.join('..', '..', 'Data', 'RawData') #Original data path
    fimeDataPath = os.path.join('..', '..', 'Data', 'FimeData') #Data path to keep merged data
    
    #name of dataste in fimeDataPath
    datasets = ['MEMEX-MCC','HVV-COVID19', 'HVV-USPOLITICS']
    dgObj = DataGathere(rawDataPath=rawDataPath, fimeDataPath=fimeDataPath, datasets=datasets)
    dgObj.getData()



