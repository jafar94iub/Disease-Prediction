import numpy as np
import pandas as pd
import pickle
import telegram.ext

# Loading the Needed Files.
descriptions = pd.read_csv("dataset/disease_description.csv")
precaution = pd.read_csv("dataset/precautions.csv")
weights = pd.read_csv('dataset/symptom_weights.csv')
specialist = pd.read_csv('dataset/specialist.csv')
weights['Symptom'] = weights['Symptom'].str.replace('_',' ')

# Loading the Model.
with open('random_forest.pkl', 'rb') as file:
    random_forest = pickle.load(file)

# Pre-processing the Data.
def preprocess(text):
    text = text.lower()
    texts = text.split(',')
    texts = [word.strip() for word in texts]
    
    textz = [0] * (17 - len(texts))
    texts.extend(textz)
    return texts

# Function to get the Disease after Prediction.
def get_disease(symp):
    a = np.array(weights["Symptom"])
    b = np.array(weights["weight"])
    for j in range(len(symp)):
        for k in range(len(a)):
            if symp[j] == a[k]:
                symp[j] = b[k]
    
    converted_symps = [symp]
    prediction = random_forest.predict(converted_symps)
    predicted_disease = prediction[0].strip()
    return predicted_disease
    
# Function to get Disease Discription.
def get_description(disease):
    description = descriptions[descriptions['Disease'] == disease]
    description = description.values[0][1]
    return description

# Fuction to get Precautions.
def get_precaution(disease):
    precautions_list = precaution.loc[precaution['Disease'] == disease, ['Precaution_1', 'Precaution_2', 'Precaution_3']].iloc[0].tolist()
    return precautions_list

# Fuction to get Disease specialist.
def get_specialist(disease):
    specialist_contact = specialist.loc[specialist['Disease'] == disease, 'Specialist']
    specialist_contact.reset_index(drop=True, inplace=True)
    return specialist_contact[0]

# Loadig Telegram TOKEN from file.
with open('token.txt', 'r') as f:
    TOKEN = f.read()

# Loadig the list of Symptoms from file.
with open('all_symptoms.txt', 'r') as f:
    ALL_SYMPTOMS = f.read()

# Start Command.
def start(update, context):
    update.message.reply_text("Please Enter Your Symptoms")

# Help Command.
def helps(update, context):
    update.message.reply_text(
        """
        This is a Disease Prediction bot:
        - /start - Start the bot.
        - /list_symptoms - List all the available Symptoms.
        """
    )

def list_symptoms(update, context):
    update.message.reply_text(ALL_SYMPTOMS)

# Message Handler.
def handle_message(update, context):
    text = update.message.text
    processed_text = preprocess(text)
    disease = get_disease(processed_text)
    description = get_description(disease)
    precautions = get_precaution(disease)
    specialists = get_specialist(disease)

    update.message.reply_text(f'Detected Disease: {disease}')
    update.message.reply_text(f'Discription: , {description}')

    update.message.reply_text(
        f"""
        Recommended things to do at Home:
        - {precautions[0].title()}
        - {precautions[1].title()}
        - {precautions[2].title()}
        """
    )
    update.message.reply_text(f'You should contact: {specialists}')

updater = telegram.ext.Updater(TOKEN, use_context = True)
dispatch = updater.dispatcher

dispatch.add_handler(telegram.ext.CommandHandler('start', start))
dispatch.add_handler(telegram.ext.CommandHandler('help', helps))
dispatch.add_handler(telegram.ext.CommandHandler('list_symptoms', list_symptoms))
dispatch.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.text, handle_message))

updater.start_polling()
updater.idle()