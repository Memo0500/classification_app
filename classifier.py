import numpy as np
import pandas as pd
import pickle as pk
import xgboost as xg


def prepare_data(form):
    lab_procedures = form.get('num_lab_procedures', 0)
    insulin = form.get('insulin',0)
    inpatient = form.get('inpatient',1)
    medications = form.get('medications',1)
    medication_change = form.get('change',0)
    time_in_hospital = form.get('time_in_hospital', 1)
    number_diagnoses = form.get('number_diagnoses', 1)
    num_procedures = form.get('num_procedures', 6)

    one_instance = {
        "change": int(medication_change),
        "num_lab_procedures": int(lab_procedures),
        "insulin": int(insulin),
        "number_inpatient": int(inpatient),
        "num_medications": int(medications),
        "age": 8,
        "time_in_hospital": int(time_in_hospital),
        "number_diagnoses": int(number_diagnoses),
        "num_procedures": int(num_procedures)
    }

    one_instance = pd.DataFrame.from_records([one_instance])

    return one_instance


def load_model():
   try:
       with open('application.model.model', 'rb') as reader:
           model = pk.load(reader)
       return model
   except Exception as e:
       print(e)
       return None


def predict(oneRecord):
    model = load_model()
    result = model.predict(oneRecord)
    return result
