import pandas as pd
import joblib

model = joblib.load('models/rf_best_model.pkl')

def predict_fate(pclass, sex, age, fare, have_cabin, family_size, title, embarked):
    sex_map ={'male': 0, 'female':1}

    embarked_q = 1 if embarked == 'Q' else 0
    embarked_s = 1 if embarked == 'S' else 0

    passengers_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_map[sex]],
        'Age': [age],
        'Fare':[fare],
        'Cabin': [have_cabin],
        'Family_size': [family_size],
        'Title': [title],
        'Embarked_Q': [embarked_q],
        'Embarked_S': [embarked_s]
    })

    prediction = model.predict(passengers_data)[0]
    probability = model.predict_proba(passengers_data)[0]

    if prediction == 1:
        print(f"Predict: Survived (life chance: {probability[1]*100:.1f}%)")
    else :
        print(f"Predict: Not Survived (life chance: {probability[1]*100:.1f}%)")



print(f"Guessing Jack's fate (Titanic Movie)")
predict_fate(pclass=3, sex='male', age=20, fare=8.0, have_cabin=0, family_size=1, title=1, embarked='S')
print(f"Guessing Rose's fate (Titanic Movie)")
predict_fate(pclass=1, sex='female', age=17, fare=100.0, have_cabin=1, family_size=2, title=2, embarked='S')
