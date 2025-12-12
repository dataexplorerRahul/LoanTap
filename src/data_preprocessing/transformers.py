import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder, WOEEncoder
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Custom Data-cleaner transformer
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self # nothing to fit, return self
    
    def transform(self, X):
        # Ensure X is a dataframe to access columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas Dataframe object")
        
        X = X.copy()

        # For object-dtype columns, strip any whitespaces in their values
        for feat in X.columns:
            if X[feat].dtype=="object":
                X[feat] = X[feat].str.strip()

        # Converting issue_d & earliest_cr_line to datetime type
        for feat in ["issue_d", "earliest_cr_line"]:    
            X[feat] = pd.to_datetime(X[feat], format="mixed")

        # Extract state code from address feature
        X["address"] = X["address"].str.extract(r'.\s([\w]{2})\s\d{4,5}$')[0]

        # Merge categories to reduce cardinality
        X["home_ownership"] = X["home_ownership"].replace(["ANY", "NONE"], "OTHER") # Merging ANY & NONE into OTHER
        X["verification_status"] = X["verification_status"].replace("Source Verified", "Verified") # Merging Source Verified into Verified
        X["application_type"] = X["application_type"].replace(["JOINT", "DIRECT_PAY"], "NON_INDIVIDUAL")
        
        return X


# Custom Feature-engineering transformer
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self # nothing to fit, return self
    
    def transform(self, X):
        # Ensure X is a dataframe to access columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas Dataframe object")
        
        X = X.copy()

        # Loan to income ratio
        X["loan_income_ratio"] = (X["loan_amnt"] / X["annual_inc"]).round(2)
        # EMI to monthly income ratio
        X["emi_ratio"] = (X["installment"] / (X["annual_inc"]/12)).round(2)
        # Credit-line age in years
        X["credit_age_years"] = ((X["issue_d"] - X["earliest_cr_line"]).dt.days / 365).round(1)
        # Total closed accounts
        X["closed_acc"] = X["total_acc"] - X["open_acc"]
        # Has negative records
        X["negative_rec"] = ((X["pub_rec"]> 0) | (X["pub_rec_bankruptcies"]>0)).astype("int")
        # Credit utilization ratio
        X["credit_util_ratio"] = (X["revol_bal"] / X["annual_inc"]).round(2)
        # Mortgage accounts ratio
        X["mortgage_ratio"] = (X["mort_acc"] / X["total_acc"]).round(2)

        return X
    
# Custom Outlier handler transformer
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        # Ensure X is a dataframe to access columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas Dataframe object")
        
        # Calculating the statistics
        self.bounds_ = {}
        for col in self.features:
            upper_bound = X[col].quantile(0.98)
            self.bounds_[col] = upper_bound
            
        return self
    
    def transform(self, X):
        # Check if fitted
        if not self.bounds_:
            raise RuntimeError("You must run fit() before transform()")
        
        X = X.copy()
        # Capping the outliers on the upper-end
        for col in self.features:
            upper_bound = self.bounds_[col]
            X[col] = X[col].clip(upper=upper_bound)
        return X
    
# Custom Feature dropper transformer
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self # Nothing to fit, return self
    
    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.features, errors="ignore")
    
# Custom Imputer transformer
class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, use_knn_imputation=False, num_features=None, cat_features=None):
        self.use_knn_imputation = use_knn_imputation
        self.num_features = num_features
        self.cat_features = cat_features
        
    def fit(self, X, y=None):
        # Ensure X is a dataframe to access columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas Dataframe object")

        # Instantiate the imputers
        if self.use_knn_imputation:
            self.num_imputer_ = KNNImputer()
        else:
            self.num_imputer_ = SimpleImputer(strategy="median")
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")

        # Fit the imputers
        if self.num_features:
            self.num_imputer_.fit(X[self.num_features])
        if self.cat_features:
            self.cat_imputer_.fit(X[self.cat_features])

        return self

    def transform(self, X):
        # Check if the imputers have been fitted
        try:
            check_is_fitted(self.num_imputer_)
            check_is_fitted(self.cat_imputer_)
        except NotFittedError:
            raise RuntimeError("You must run fit() before transform()")

        X = X.copy()
        # Tranform numerical features
        if self.num_features:
            imputed_nums = self.num_imputer_.transform(X[self.num_features])
            X[self.num_features] = imputed_nums

        # Transform categorical features
        if self.cat_features:
            imputed_cats = self.cat_imputer_.transform(X[self.cat_features])
            X[self.cat_features] = imputed_cats

        return X
    

# Custom Categorical Encoder transformer
class CatEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_type="woe", ohe_features=None, supervised_features=None):
        self.encoder_type = encoder_type
        self.ohe_features = ohe_features
        self.supervised_features = supervised_features

        # Ordinal categories mapping
        self.grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        self.sub_grade_map = {"A1": 0, "A2": 1, "A3": 2, "A4": 3, "A5": 4, "B1": 5, "B2": 6, "B3": 7, "B4": 8, "B5": 9, "C1": 10, "C2": 11, "C3": 12, "C4": 13, "C5": 14, "D1": 15, "D2": 16, "D3": 17, "D4": 18, "D5": 19, "E1": 20, "E2": 21, "E3": 22, "E4": 23, "E5": 24, "F1": 25, "F2": 26, "F3": 27, "F4": 28, "F5": 29, "G1": 30, "G2": 31, "G3": 32, "G4": 33, "G5": 34}
        self.emp_length_map = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,  "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}

    def fit(self, X, y=None):
        # Ensure X is a dataframe to access columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas Dataframe object")
        
        # Instantiate the encoder
        if self.encoder_type == "woe":
            self.sup_encoder_ = WOEEncoder(cols=self.supervised_features)
        elif self.encoder_type == "target":
            self.sup_encoder_ = TargetEncoder(cols=self.supervised_features)
        else:
            raise ValueError("encoder_type must be either 'woe' or 'target'") 
        
        # Fit the encoders
        if self.supervised_features:
            self.sup_encoder_.fit(X, y)

        if self.ohe_features:
            self.ohe_encoder_ = OneHotEncoder(drop="first", handle_unknown="ignore")
            self.ohe_encoder_.fit(X[self.ohe_features])
        
        return self

    def transform(self, X):
        # Check if the encoder has fitted
        try:
            check_is_fitted(self.sup_encoder_)
            check_is_fitted(self.ohe_encoder_)
        except NotFittedError:
            raise RuntimeError("You must run fit() before transform()")
        
        X = X.copy()

        # Supervised categorical encoding (WOE or Target)
        if self.supervised_features:
            X = self.sup_encoder_.transform(X)

        # One-hot encoding
        if self.ohe_features:
            ohe_feature_names = self.ohe_encoder_.get_feature_names_out()
            X_ohe = pd.DataFrame(data=self.ohe_encoder_.transform(X[self.ohe_features]).toarray(), 
                                 columns=ohe_feature_names,
                                 index=X.index)
            X = X.drop(columns=self.ohe_features, errors="ignore") # drop the original ohe_features
            X = pd.concat([X, X_ohe], axis=1) # concat transformed ohe_features to dataset
        
        # Ordinal encoding
        X["grade"] = X["grade"].map(self.grade_map)
        X["sub_grade"] = X["sub_grade"].map(self.sub_grade_map)
        X["emp_length"] = X["emp_length"].map(self.emp_length_map)

        return X
    

# Custom Scaler transformer
class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        # Ensure X is a dataframe to access columns
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas Dataframe object")
        
        # Instantiate the scaler
        self.scaler_ = RobustScaler()

        # Fit the scaler
        if self.features:
            self.scaler_.fit(X[self.features])

        return self
    
    def transform(self, X):
        # Check if the scaler has fitted
        try:
            check_is_fitted(self.scaler_)
        except NotFittedError:
            raise RuntimeError("You must run fit() before transform()")
        
        X = X.copy()

        # Scale the numerical features
        if self.features:
            scaled_nums = self.scaler_.transform(X[self.features])
            X[self.features] = scaled_nums

        return X