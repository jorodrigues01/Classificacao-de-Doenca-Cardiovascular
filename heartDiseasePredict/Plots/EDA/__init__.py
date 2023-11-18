import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    def __init__(self, dataframe):
        self.df = dataframe

    def boxplot(self):
        fig = plt.figure(figsize=(10,7))

        linear_columns = self.df.drop(columns=['Gender', 'Class'])
        plt.boxplot(linear_columns)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.FixedFormatter(linear_columns.columns))

        plt.show()

    def Gender_col(self):
        neg = self.df[self.df.Class == 0]
        pos = self.df[self.df.Class == 1]

        fig, axs = plt.subplots(figsize=(10, 6), constrained_layout=True)
        fig.suptitle('Column: Gender')
        # 0: Women  /   1: Men'

        cmap = plt.get_cmap('Blues')
        colors = cmap([0.3, 0.6, 0.9])

        counts, edges, bars = axs.hist([neg.Gender, pos.Gender, self.df.Gender], bins=2,
                                       label=['Neg. samples', 'Pos. samples',
                                              'Total observations'],
                                       color=colors)

        for b in bars:
            axs.bar_label(b)

        axs.xaxis.set_major_locator(plt.FixedLocator([.25, .75]))
        axs.xaxis.set_major_formatter(plt.FixedFormatter(['Female', 'Male']))
        axs.legend(title='Legend', loc=(1.02, .5))

        plt.show()


    def Age_col(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle('Column: Age')

        sns.histplot(data=self.df, x=self.df.Age, hue='Class', element="step", common_norm=False,
                     ax=axs[0], bins=10)
        axs[0].set_xlabel('Age (Years)')

        sns.histplot(self.df.Age, element='step', ax=axs[1], bins=10)
        axs[1].set_xlabel('Age (Years)')

        sns.boxplot(data=self.df, x='Class', y=self.df.Age, ax=axs[2])
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))
        axs[2].set_ylabel('Age (Years)')

        plt.show()

    def HeartRate_col(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle('Column: Heart Rate (Impulse)')

        sns.histplot(data=self.df, x=self.df['Heart_Rate'], hue='Class', element="step",
                     common_norm=False, ax=axs[0])
        axs[0].set_xlabel('Heart rate (BPM)')

        sns.histplot(self.df['Heart_Rate'], element="step", ax=axs[1])
        axs[1].set_xlabel('Heart rate (BPM)')

        sns.boxplot(data=self.df, x='Class', y=self.df['Heart_Rate'], ax=axs[2])
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))
        axs[2].set_ylabel('Heart rate (BPM)')

        plt.show()

    def PressureHigh_col(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle('Column: Pressure_High / Systolic BP')

        sns.histplot(data=self.df, x=self.df['Pressure_High'], hue='Class', element="step",
                     common_norm=False, ax=axs[0])
        axs[0].set_xlabel('Blood Pressure (mmHg)')

        sns.histplot(self.df['Pressure_High'], element="step", ax=axs[1])
        axs[1].set_xlabel('Blood Pressure (mmHg)')

        sns.boxplot(data=self.df, x='Class', y=self.df['Pressure_High'], ax=axs[2])
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))
        axs[2].set_ylabel('Blood Pressure (mmHg)')

        plt.show()

    def PressureLow_col(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle('Column: Pressure_Low / Diastolic BP')

        sns.histplot(data=self.df, x=self.df['Pressure_Low'], hue='Class', element="step",
                     common_norm=False, ax=axs[0])
        axs[0].set_xlabel('Blood Pressure (mmHg)')

        sns.histplot(self.df['Pressure_Low'], element="step", ax=axs[1])
        axs[1].set_xlabel('Blood Pressure (mmHg)')

        sns.boxplot(data=self.df, x='Class', y=self.df['Pressure_Low'], ax=axs[2])
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))
        axs[2].set_ylabel('Blood Pressure (mmHg)')

        plt.show()

    def Glucose_col(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle('Column: Glucose')

        sns.histplot(data=self.df, x=self.df.Glucose, hue='Class', element="step",
                     common_norm=False, ax=axs[0])
        axs[0].set_xlabel('Glucose (mg/dl)')

        sns.histplot(self.df.Glucose, element='step', ax=axs[1])
        axs[1].set_xlabel('Glucose (mg/dl)')

        sns.boxplot(data=self.df, x='Class', y=self.df.Glucose, ax=axs[2])
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))
        axs[2].set_ylabel('Glucose (mg/dl)')

        plt.show()

    def CK_MB_col(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle('Column: CK MB')

        sns.histplot(data=self.df, x=self.df['CK_MB'], hue='Class', element="step",
                     common_norm=False, ax=axs[0])
        axs[0].set_xlabel('CK-MB (ng/ml)')

        sns.histplot(self.df['CK_MB'], element="step", ax=axs[1])
        axs[1].set_xlabel('CK-MB (ng/ml)')

        sns.boxplot(data=self.df, x='Class', y=self.df['CK_MB'], ax=axs[2])
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))
        axs[2].set_ylabel('CK-MB (ng/ml)')

        plt.show()

    def Troponin_col(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        fig.suptitle('Column: Troponin')

        sns.histplot(data=self.df, x=self.df.Troponin, hue='Class', element="step",
                     common_norm=False, ax=axs[0])
        axs[0].set_xlabel('Troponin (ng/ml)')

        sns.histplot(self.df.Troponin, element='step', ax=axs[1])
        axs[1].set_xlabel('Troponin (ng/ml)')

        sns.boxplot(data=self.df, x='Class', y=self.df.Troponin, ax=axs[2])
        axs[2].xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))
        axs[2].set_ylabel('Troponin (ng/ml)')

        plt.show()

    def get_limits(self):
        q1 = self.df.quantile(.25)
        q3 = self.df.quantile(.75)

        iqr = q3 - q1

        lim_sup = q3 + 1.5 * iqr
        lim_inf = q1 - 1.5 * iqr

        return [lim_sup.tolist(), lim_inf.tolist()]

    def drop_CKMB_Troponin_Outliers(self):
        limits = self.get_limits()
        self.df = self.df[(self.df.CK_MB < limits[0][6]) & (self.df.Troponin < limits[0][7])]


    def correlationMatrix(self):
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(8, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='Reds', cbar=True)
        plt.title('Correlation Matrix')
        plt.show();
