# Imorting Libraries
import pickle
import numpy as np
import pandas as pd

# Loading the Needed Files
descriptions = pd.read_csv("dataset/disease_description.csv")
precaution = pd.read_csv("dataset/precautions.csv")
weights = pd.read_csv('dataset/symptom_weights.csv')
specialist = pd.read_csv('dataset/specialist.csv')
weights['Symptom'] = weights['Symptom'].str.replace('_',' ')

# Loading The Model
with open('random_forest.pkl', 'rb') as file:
    random_forest = pickle.load(file)

# Pre-processing the Data
def preprocess(text):
    text = text.lower()
    texts = text.split(',')
    texts = [word.strip() for word in texts]
    
    textz = [0] * (17 - len(texts))
    texts.extend(textz)
    return texts

# Function for Prediction
def prediction(model, symp):
    print(symp)
    a = np.array(weights["Symptom"])
    b = np.array(weights["weight"])
    for j in range(len(symp)):
        for k in range(len(a)):
            if symp[j] == a[k]:
                symp[j] = b[k]
    
    converted_symps = [symp]
    prediction = model.predict(converted_symps)
    predicted_disease = prediction[0].strip()
    
    description = descriptions[descriptions['Disease'] == prediction[0]]
    description = description.values[0][1]

    precautions_list = precaution.loc[precaution['Disease'] == predicted_disease, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].iloc[0].tolist()
    print("Precaution List: ", precautions_list)
    print("Disease Name: ", prediction[0])
    print("Discription: ", description)

    print("Recommended Things to do at Home: ")
    for i in precautions_list:
        print(i.title())
    
    specialist_contact = specialist.loc[specialist['Disease'] == predicted_disease, 'Specialist']
    specialist_contact.reset_index(drop=True, inplace=True)
    print(f"You Should Contact: {specialist_contact[0]}")