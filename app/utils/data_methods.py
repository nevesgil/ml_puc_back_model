import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class DataReader:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_csv(self):
        df = pd.read_csv(self.filepath)
        return df


class NullRemover:
    def __init__(self, df):
        """
        Initializes the NullRemover class with a DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame that may contain null values.
        """
        self.df = df
        self.categorical_variables = self.df.select_dtypes(include=["object"]).columns
        self.numerical_variables = self.df._get_numeric_data().columns

    def remove_nulls(self):
        """
        Removes null values from both categorical and numerical columns.

        Categorical columns are filled with the mode (most frequent value),
        and numerical columns are filled with the mean.
        """
        # Handle categorical variables
        for cat_col in self.categorical_variables:
            if cat_col != "Name":
                self.df[cat_col] = self.df[cat_col].fillna(self.df[cat_col].mode()[0])

        # Handle numerical variables
        for num_col in self.numerical_variables:
            self.df[num_col] = self.df[num_col].fillna(self.df[num_col].mean())

        return self.df

    def get_nulls_percentage(self):
        """
        Returns the percentage of null values in each column of the DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing the percentage of null values in each column.
        """
        dict_nulls = {}
        for col in self.df.columns:
            percentage_null_values = (
                str(round(self.df[col].isnull().sum() / len(self.df), 2)) + "%"
            )
            dict_nulls[col] = percentage_null_values

        df_nulls = pd.DataFrame(
            data=list(dict_nulls.values()),
            index=list(dict_nulls.keys()),
            columns=["% nulls"],
        )
        return df_nulls


class DataTransformer:
    def __init__(self, df):
        """
        Initializes the DataTransformer class with a DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame for data transformation.
        """
        self.df = df

    def sum_spending_columns(self):
        """
        Sums up the spending categories (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)
        and creates a new 'TotalSpend' column. The original spending columns are dropped.

        Returns:
        pd.DataFrame: The transformed DataFrame with 'TotalSpend' and without individual spending columns.
        """
        # Summing up the spending categories
        self.df["TotalSpend"] = self.df[
            ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        ].sum(axis=1)
        # Dropping the individual spending columns
        self.df = self.df.drop(
            ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], axis=1
        )
        return self.df

    def map_categorical_columns(self):
        """
        Maps categorical values in 'HomePlanet' and 'Destination' columns to integers.

        HomePlanet: {'Earth':1, 'Europa':2, 'Mars':3}
        Destination: {'TRAPPIST-1e':1, 'PSO J318.5-22':2, '55 Cancri e':3}

        # TODO: map this dynamically in a database

        Returns:
        pd.DataFrame: The transformed DataFrame with mapped values for 'HomePlanet' and 'Destination'.
        """
        # Mapping values for HomePlanet and Destination
        self.df["HomePlanet"] = (
            self.df["HomePlanet"].map({"Earth": 1, "Europa": 2, "Mars": 3}).astype(int)
        )
        self.df["Destination"] = (
            self.df["Destination"]
            .map({"TRAPPIST-1e": 1, "PSO J318.5-22": 2, "55 Cancri e": 3})
            .astype(int)
        )
        return self.df

    def drop_columns(self):
        """
        Drops the 'Name', 'Cabin', and 'PassengerID' columns from the DataFrame.

        Returns:
        pd.DataFrame: The transformed DataFrame without the specified columns.
        """
        self.df = self.df.drop(["Name", "Cabin", "PassengerId"], axis=1)
        return self.df

    def convert_bool_columns(self):
        """
        Converts the 'CryoSleep' and 'VIP' columns to boolean type.

        Returns:
        pd.DataFrame: The transformed DataFrame with 'CryoSleep' and 'VIP' as boolean.
        """
        self.df["CryoSleep"] = self.df["CryoSleep"].astype(int)
        self.df["VIP"] = self.df["VIP"].astype(int)
        return self.df


class BaseEvaluator:
    def __init__(self, X_train, y_train, kfold, scoring):
        """
        Initialize the BaseEvaluator with training data, cross-validation method, and scoring function.

        :param X_train: Training features
        :param y_train: Training labels
        :param kfold: Cross-validation splitting strategy
        :param scoring: Scoring method to evaluate the models
        """
        self.X_train = X_train
        self.y_train = y_train
        self.kfold = kfold
        self.scoring = scoring
        self.models = []
        self.results = []
        self.names = []

    def create_models(self):
        """
        Create a list of machine learning models to be evaluated.
        """
        self.models.append(("KNN", KNeighborsClassifier()))
        self.models.append(("CART", DecisionTreeClassifier()))
        self.models.append(("NB", GaussianNB()))
        self.models.append(("SVM", SVC()))

    def evaluate_models(self):
        """
        Evaluate each model using cross-validation and store the results.
        """
        for name, model in self.models:
            cv_results = cross_val_score(
                model, self.X_train, self.y_train, cv=self.kfold, scoring=self.scoring
            )
            self.results.append(cv_results)
            self.names.append(name)
            print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

    def plot_results(self):
        """
        Plot the comparison of models using a boxplot.
        """
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle("Model Comparison")
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names)
        plt.show()


class ScaleEvaluator:
    def __init__(self, X_train, y_train, kfold, scoring):
        """
        Initialize the ScaleEvaluator with training data, cross-validation method, and scoring function.

        :param X_train: Training features
        :param y_train: Training labels
        :param kfold: Cross-validation splitting strategy
        :param scoring: Scoring method to evaluate the models
        """
        self.X_train = X_train
        self.y_train = y_train
        self.kfold = kfold
        self.scoring = scoring
        self.pipelines = []
        self.results = []
        self.names = []

    def create_pipelines(self):
        """
        Create pipelines for various models with original data, standardized data, and normalized data.
        """
        # Define the models
        knn = ("KNN", KNeighborsClassifier())
        cart = ("CART", DecisionTreeClassifier())
        naive_bayes = ("NB", GaussianNB())
        svm = ("SVM", SVC())

        # Define the transformations
        standard_scaler = ("StandardScaler", StandardScaler())
        min_max_scaler = ("MinMaxScaler", MinMaxScaler())

        # Original dataset pipelines
        self.pipelines.append(("KNN-orig", Pipeline([knn])))
        self.pipelines.append(("CART-orig", Pipeline([cart])))
        self.pipelines.append(("NB-orig", Pipeline([naive_bayes])))
        self.pipelines.append(("SVM-orig", Pipeline([svm])))

        # Standardized dataset pipelines
        self.pipelines.append(("KNN-std", Pipeline([standard_scaler, knn])))
        self.pipelines.append(("CART-std", Pipeline([standard_scaler, cart])))
        self.pipelines.append(("NB-std", Pipeline([standard_scaler, naive_bayes])))
        self.pipelines.append(("SVM-std", Pipeline([standard_scaler, svm])))

        # Normalized dataset pipelines
        self.pipelines.append(("KNN-mm", Pipeline([min_max_scaler, knn])))
        self.pipelines.append(("CART-mm", Pipeline([min_max_scaler, cart])))
        self.pipelines.append(("NB-mm", Pipeline([min_max_scaler, naive_bayes])))
        self.pipelines.append(("SVM-mm", Pipeline([min_max_scaler, svm])))

    def evaluate_pipelines(self):
        """
        Evaluate each pipeline using cross-validation and store the results.
        """
        for name, model in self.pipelines:
            cv_results = cross_val_score(
                model, self.X_train, self.y_train, cv=self.kfold, scoring=self.scoring
            )
            self.results.append(cv_results)
            self.names.append(name)
            print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")

    def plot_results(self):
        """
        Plot the comparison of pipelines using a boxplot.
        """
        fig = plt.figure(figsize=(25, 6))
        fig.suptitle("Model Comparison - Original, Standardized, and Normalized Data")
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names, rotation=90)
        plt.show()


class StdTuning:
    def __init__(self, X_train, y_train, kfold=5, scoring="accuracy"):
        """
        Initialize the StdTuning with training data, cross-validation method, and scoring function.

        :param X_train: Training features
        :param y_train: Training labels
        :param kfold: Cross-validation splitting strategy (default=5)
        :param scoring: Scoring method to evaluate the models (default='accuracy')
        """
        self.X_train = X_train
        self.y_train = y_train
        self.kfold = kfold
        self.scoring = scoring
        self.pipelines = []
        self.param_grids = {}
        np.random.seed(7)  # Set a global seed for reproducibility

    def create_pipelines(self):
        """
        Create pipelines with standard scaling for KNN, CART, NB, and SVM models.
        """
        # Define the models
        models = [
            ("KNN", KNeighborsClassifier()),
            ("CART", DecisionTreeClassifier()),
            ("NB", GaussianNB()),
            ("SVM", SVC()),
        ]

        # Define the standard scaler step
        standard_scaler = ("StandardScaler", StandardScaler())

        # Create pipelines for each model with standard scaling
        for name, model in models:
            self.pipelines.append(
                (name + "-std", Pipeline(steps=[standard_scaler, (name, model)]))
            )

    def define_param_grids(self):
        """
        Define the parameter grids for GridSearchCV tuning for each model.
        """
        self.param_grids = {
            "KNN": {
                "KNN__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
                "KNN__metric": ["euclidean", "manhattan", "minkowski"],
                "KNN__weights": ["uniform", "distance"],
            },
            "CART": {
                "CART__max_depth": [None, 10, 20, 30, 40, 50],
                "CART__min_samples_split": [2, 5, 10],
                "CART__min_samples_leaf": [1, 2, 4],
                "CART__criterion": ["gini", "entropy"],
            },
            "NB": {"NB__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
            "SVM": {
                "SVM__C": [0.1, 1, 10],
                "SVM__gamma": ["scale", "auto"],
                "SVM__kernel": ["rbf", "linear", "poly"],
            },
        }

    def run_grid_search(self):
        """
        Run GridSearchCV for each pipeline and print the best configurations.
        """
        for name, pipeline in self.pipelines:
            model_type = name.split("-")[
                0
            ]  # Extract model type (e.g., 'KNN', 'CART', etc.)
            if model_type in self.param_grids:
                param_grid = self.param_grids[model_type]
            else:
                param_grid = {}  # Default empty param grid if not defined

            # Run GridSearchCV
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.kfold,
            )
            grid.fit(self.X_train, self.y_train)

            # Print the best score and parameters
            print(
                f"Model: {name} - Best: {grid.best_score_:.3f} using {grid.best_params_}"
            )
