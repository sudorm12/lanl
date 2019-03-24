import pandas as pd


# TODO: add another class layer to seperate functionality for time series and binary classification
# TODO: add detailed documentation on the class
class DataLoader:
    def __init__(self):
        pass

    def get_index(self):
        raise NotImplementedError

    def load_train_data(self, fit_transform=True, **loader_args):
        raise NotImplementedError

    def load_test_data(self, **loader_args):
        raise NotImplementedError

    def get_input_shape(self):
        raise NotImplementedError

    def load_train_val(self, train_args=None, val_args=None, loader_args=None):
        # replace default arguments with empty dicts to allow enumeration
        # in later function calls to work properly
        if train_args is None:
            train_args = {}
        if val_args is None:
            val_args = {}
        if loader_args is None:
            loader_args = {}

        train_data, train_target = self.load_train_data(fit_transform=True, **train_args, **loader_args)
        val_data, val_target = self.load_train_data(fit_transform=False, **val_args, **loader_args)
        return train_data, train_target, val_data, val_target

    def load_train_test(self, train_args=None, test_args=None, loader_args=None):
        # replace default arguments with empty dicts to allow enumeration
        # in later function calls to work properly
        if train_args is None:
            train_args = {}
        if test_args is None:
            test_args = {}
        if loader_args is None:
            loader_args = {}

        train_data, train_target = self.load_train_data(fit_transform=True, **train_args, **loader_args)
        test_data, test_target = self.load_test_data(**test_args, **loader_args)
        return train_data, train_target, test_data, test_target

    @staticmethod
    def _yn_cols_to_boolean(df, cols):
        yn_map = {'Y': 1,
                  'N': 0}
        return df[cols].replace(yn_map)

    @staticmethod
    def _cat_data(df):
        df_clean = df.copy()

        # detect columns with dtype 'object'
        cols = df.columns[df.dtypes == 'object']

        # fill na values with 'Unspecified'
        cat_na_count = df_clean[cols].isna().sum(axis=0)
        cat_na_map = {cat: 'Unspecified' for cat in cat_na_count[cat_na_count > 0].index}
        df_clean = df_clean.fillna(value=cat_na_map)

        # convert columns to categorical
        cat_labels = {}
        for cat_col in cols:
            cat_labels[cat_col] = df_clean[cat_col].unique()
            df_clean[cat_col] = pd.Categorical(df_clean[cat_col], categories=cat_labels[cat_col])

        return df_clean

    @staticmethod
    def _cat_data_dummies(df):
        df_clean = df.copy()

        # convert categorical columns to Categorical dtype
        df_clean = DataLoader._cat_data(df_clean)

        # convert Categorical columns to dummy columns
        df_clean = pd.get_dummies(df_clean)

        # return dataframe with dummy columns
        return df_clean
