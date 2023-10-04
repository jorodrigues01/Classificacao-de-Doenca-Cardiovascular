import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import Perceptron


def one_hot_encodingLabel(val):
    if val != 'positive':
        return 0
    return 1


def accuracy(target, obtained):
    acc = sum(target == obtained) / len(target)
    return acc


def main():
    url = "https://raw.githubusercontent.com/jorodrigues01/Classificacao-de-Doenca-Cardiovascular/main/HeartAttack.csv"
    df = pd.read_csv(url)

    df.rename(columns={'age': 'Age', 'gender': 'Gender', 'impluse': 'Heart rate (Impulse)',
                       'pressurehight': 'Systolic BP (Pressure High)',
                       'pressurelow': 'Pressure Low (Diastolic BP)', 'glucose': 'Glucose', 'kcm': 'CK MB',
                       'troponin': 'Troponin', 'class': 'Class'},
              inplace=True)

    df.Class = df['Class'].apply(one_hot_encodingLabel)

    label = df.Class
    features = df.drop(columns='Class')

    X_train, X_test, y_train, y_test = train_test_split(features, label, random_state=42, test_size=0.25)

    p = Perceptron(eta=0.01, threshold=0.2, n_epochs=300)

    p.train(X_train, y_train)
    predict = p.test(X_test)

    print(f'\n\n Comparing the labels with the predictions obtained by the neural network, we got a '
          f'{accuracy(y_test, predict)} of accuracy! ')


if __name__ == '__main__':
    main()
