import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Plots.EDA import EDA
from Plots.Accuracies import knnScores, classifiersAccuracies
from LogisticReg import LogisticReg
from DTree.DecisionTree import DecisionTree
from KNN import KNN


def accuracy(target, obtained):
  score = sum(target == obtained)/ len(target)
  return score

def z_score(data):
  #######   Z-Score FORMULA   ########
  #                                  #
  #      z' = (x-mean)/std_dev       #
  #                                  #
  ####################################
  return (data - np.mean(data, axis=0))/np.std(data, axis=0)

def get_Kscores(X_train, y_train, X_test, y_test):
    knn_accuracies = []
    k_numbers = np.arange(1, 16, 2)

    for k in k_numbers:
        knn = KNN(X_train, y_train, k)
        predict = knn.test(X_test)

        knn_accuracies.append(accuracy(y_test, predict))

    return pd.Series(knn_accuracies, index=k_numbers) * 100


def main():
    url = "https://raw.githubusercontent.com/jorodrigues01/Classificacao-de-Doenca-Cardiovascular/main/HeartAttack.csv"
    df = pd.read_csv(url)

    print(df.info())

    df.rename(columns={'age': 'Age',
                       'gender': 'Gender',
                       'impluse': 'Heart_Rate',
                       'pressurehight': 'Pressure_High',
                       'pressurelow': 'Pressure_Low',
                       'glucose': 'Glucose',
                       'kcm': 'CK_MB',
                       'troponin': 'Troponin',
                       'class': 'Class'}, inplace=True)

    print('\n ====== MISSING VALUES ======')
    print(f'{df.isna().sum().sum()} observations got missing values')
    print('By column:')
    print(df.isna().sum())

    df.Class = np.where(df.Class == 'positive', 1, 0)

    # Exploratory Data Analysis

    df.groupby('Class').mean()

    eda = EDA(df)

    eda.boxplot()

    eda.df = df[df['Heart_Rate'] < 1000].reset_index(drop=True)

    eda.Gender_col()
    eda.Age_col()
    eda.HeartRate_col()
    eda.PressureHigh_col()
    eda.PressureLow_col()
    eda.Glucose_col()
    eda.CK_MB_col()
    eda.Troponin_col()

    eda.drop_CKMB_Troponin_Outliers()
    df = eda.df

    eda.correlationMatrix()

    # Classifiers

    label = df.Class
    features = df.drop(columns='Class')

    X_train, X_test, y_train, y_test = train_test_split(features, label, random_state=42, test_size=0.25)


    ### Logistic Regression
    print('\n\n ### Logistic Regression ###')
    logReg = LogisticReg(n_epochs=2000)
    logReg.train(X_train, y_train)
    LR_prediction = logReg.test(X_test)
    LR_score = accuracy(y_test, LR_prediction)

    print(f'Comparing the labels with the predictions obtained by the Logistic regression algorithm, '
          f'we got a {LR_score} of accuracy! ')


    ### Decision Tree
    print('\n\n ### Decision Tree ###')
    decTree = DecisionTree(min_split=2, max_depth=100)
    decTree.fit(X_train, y_train)
    DT_prediction = decTree.predict(X_test)
    DT_score = accuracy(y_test, DT_prediction)

    print(f'Comparing the labels with the predictions obtained by the Decision Tree algorithm, '
          f'we got a {DT_score} of accuracy! ')

    ### KNN algorithm
    print('\n\n ### KNN algorithm ###')
    knn = KNN(X_train, y_train, k=3)
    KNN_prediction = knn.test(X_test)
    KNN_score = accuracy(y_test, KNN_prediction)

    print(f'Comparing the labels with the predictions obtained by the kNN algorithm, '
          f'we got a {KNN_score} of accuracy! ')

    knn_accuracies = get_Kscores(X_train, y_train, X_test, y_test)

    knnScores(knn_accuracies)

    classifiersAccuracies(LR_score, DT_score, KNN_score)


if __name__ == '__main__':
    main()
