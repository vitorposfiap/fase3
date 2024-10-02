import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from lib.DataInput import AdvancedImputer


class RegressionDataAnalyzer:
    def __init__(self, dataframe: pd.DataFrame, target_variable: str):
        self.df = dataframe
        self.target_variable = target_variable
        self.variable_types = None

    def analyze_variable_types(self):
        """Analisar e retornar tipos de variáveis (contínuas, categóricas, binárias) e seus tipos de dados."""
        variable_types = {}
        for column in self.df.columns:
            dtype = self.df[column].dtype

            if column == self.target_variable:
                variable_types[column] = {'type': 'dependent', 'data_type': 'continuous'}
            elif np.issubdtype(dtype, np.number):
                if len(self.df[column].unique()) == 2:
                    variable_types[column] = {'type': 'binary', 'data_type': 'numeric'}
                elif len(self.df[column].unique()) > 15:
                    variable_types[column] = {'type': 'continuous', 'data_type': 'numeric'}
                else:
                    variable_types[column] = {'type': 'categorical', 'data_type': 'numeric'}
            elif np.issubdtype(dtype, np.object_):
                variable_types[column] = {'type': 'categorical', 'data_type': 'text'}
            elif np.issubdtype(dtype, np.bool_):
                variable_types[column] = {'type': 'binary', 'data_type': 'boolean'}
            else:
                variable_types[column] = {'type': 'unknown', 'data_type': str(dtype)}

        self.variable_types = variable_types
        return variable_types

    def perform_data_analysis(self):
        """Executar análise de dados com base nos tipos de variáveis."""
        analysis_results = {}

        for column, details in self.variable_types.items():
            if details['type'] == 'continuous':
                analysis_results[column] = {
                    'missing_values': self.df[column].isnull().sum(),
                    'mean': self.df[column].mean(),
                    'std_dev': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max()
                }
            elif details['type'] in ['categorical', 'binary']:
                analysis_results[column] = {
                    'missing_values': self.df[column].isnull().sum(),
                    'unique_values': self.df[column].nunique(),
                    'top_value': self.df[column].value_counts().idxmax(),
                    'top_value_freq': self.df[column].value_counts().max()
                }

        return analysis_results

    def identify_issues(self):
        """Identificar os principais fatores prejudiciais para um modelo de regressão."""
        issues = {}

        for column, details in self.variable_types.items():
            if details['type'] in ['continuous', 'binary']:
                if self.df[column].isnull().sum() > 0:
                    issues[column] = 'missing_values'

                if self.df[column].nunique() == 1:
                    issues[column] = 'low_variance'

            elif details['type'] == 'categorical':
                if self.df[column].isnull().sum() > 0:
                    issues[column] = 'missing_values'

                if self.df[column].nunique() > 50:
                    issues[column] = 'high_cardinality'

        return issues

    def treat_issues(self):
        """Tratar problemas identificados e retornar dataframe pronto para o modelo de regressão."""
        issues = self.identify_issues()
        treated_issues = []
        label_encoders = {}

        # Preenchimento avançado de valores nulos usando a biblioteca importada
        if 'missing_values' in issues.values():
            imputer = AdvancedImputer(self.df)
            self.df = imputer.knn_impute()
            treated_issues.append('missing_values tratado com KNN Imputer')

        for column, issue in issues.items():
            if issue == 'low_variance':
                self.df.drop(columns=[column], inplace=True)
                treated_issues.append(f'{column}: baixa variância removida')

            elif issue == 'high_cardinality':
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                label_encoders[column] = le
                treated_issues.append(f'{column}: alta cardinalidade tratada com LabelEncoder')

        for column, details in self.variable_types.items():
            if details['data_type'] == 'text' and details['type'] == 'categorical':
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                label_encoders[column] = le
                treated_issues.append(f'{column}: texto categórico codificado com LabelEncoder')

        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        f.close()
        
        print("Problemas tratados:")
        for issue in treated_issues:
            print(issue)

        return self.df

    def plot_variable_distributions(self):
        """Plotar gráfico de distribuição das variáveis do dataframe."""
        for column in self.df.columns:
            plt.figure(figsize=(10, 5))
            if self.variable_types[column]['type'] == 'continuous':
                self.df[column].plot(kind='hist', bins=30, alpha=0.7)
                plt.title(f'Distribuição de {column}')
                plt.xlabel(column)
                plt.ylabel('Frequência')
            elif self.variable_types[column]['type'] in ['categorical', 'binary']:
                self.df[column].value_counts().plot(kind='bar', alpha=0.7)
                plt.title(f'Distribuição de {column}')
                plt.xlabel(column)
                plt.ylabel('Contagem')
            plt.grid(axis='y', linestyle='--')
            plt.show()

    def analyze_correlations(self):
        """Gerar análise de correlação entre as variáveis independentes e a variável dependente, e plotar gráficos."""
        correlation_results = {}
        target = self.df[self.target_variable]

        for column, details in self.variable_types.items():
            if column == self.target_variable:
                continue

            plt.figure(figsize=(10, 5))
            if details['type'] == 'continuous':
                corr, _ = pearsonr(self.df[column].dropna().astype(float), target.dropna().astype(float))
                correlation_results[column] = {'correlation': corr, 'method': 'Pearson'}
                plt.scatter(self.df[column], target, alpha=0.7)
                plt.title(f'Correlação entre {column} e {self.target_variable} (Pearson: {corr:.2f})')
                plt.xlabel(column)
                plt.ylabel(self.target_variable)
            elif details['type'] == 'binary':
                corr, _ = pointbiserialr(self.df[column].dropna().astype(float), target.dropna().astype(float))
                correlation_results[column] = {'correlation': corr, 'method': 'Point-Biserial'}
                plt.scatter(self.df[column], target, alpha=0.7)
                plt.title(f'Correlação entre {column} e {self.target_variable} (Point-Biserial: {corr:.2f})')
                plt.xlabel(column)
                plt.ylabel(self.target_variable)
            elif details['type'] == 'categorical':
                le = LabelEncoder()
                encoded_column = le.fit_transform(self.df[column].dropna())
                corr, _ = spearmanr(encoded_column, target[self.df[column].notna()])
                correlation_results[column] = {'correlation': corr, 'method': 'Spearman'}
                plt.scatter(encoded_column, target[self.df[column].notna()], alpha=0.7)
                plt.title(f'Correlação entre {column} e {self.target_variable} (Spearman: {corr:.2f})')
                plt.xlabel(column)
                plt.ylabel(self.target_variable)

            plt.grid(axis='y', linestyle='--')
            plt.show()

        return correlation_results