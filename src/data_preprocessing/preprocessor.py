from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.data_preprocessing.transformers import DataCleaner, Imputer, OutlierHandler, FeatureEngineer, FeatureDropper, CatEncoder, Scaler, DtypeConverter

# Build pipeline
def build_pipeline(
        numerical_features, 
        engineered_features,
        categorical_features,
        supervised_features,
        onehot_features,
        ordinal_features,
        use_imputation=True, 
        use_outlier_capping=True, 
        use_encoding=True,
        use_scaling=True,
        use_smote=False, 
        features_to_drop=None,
        convert_cat_dtype=False
    ):
    # Feature types
    scaling_features = numerical_features + engineered_features + ordinal_features # including ordinal-encoded except 

    #-----Imputation-----
    if use_imputation:
        imputer = Imputer(use_knn_imputation=False, 
                          num_features=numerical_features,
                          cat_features=categorical_features)
    else:
        imputer = "passthrough"

    #-----Outlier handling-----
    if use_outlier_capping:
        outlier_capper = OutlierHandler(features=numerical_features)
    else:
        outlier_capper = "passthrough"

    #-----Categorical encoding-----
    if use_encoding:
        cat_encoder = CatEncoder(ohe_features=onehot_features, supervised_features=supervised_features)
    else:
        cat_encoder = "passthrough"
    
    #-----Scaling-----
    if use_scaling:
        scaler = Scaler(features=scaling_features)
    else:
        scaler = "passthrough"
        
    #-----Dropping specified features-----
    if features_to_drop:
        feat_dropper = FeatureDropper(features=features_to_drop)
    else:
        feat_dropper = "passthrough"

    #-----Oversampling-----
    if use_smote:
        oversampler = SMOTE()
    else:
        oversampler = "passthrough"

    #-----Converting Categorical features' dtype-----
    # Useful for algorithms like Xgboost to handle categorical features
    if convert_cat_dtype:
        dtype_converter = DtypeConverter()
    else:
        dtype_converter = "passthrough"

    # Preprocessing pipeline
    final_pipeline = Pipeline(steps=[
        ("data_cleaner", DataCleaner()),
        ("imputer", imputer),
        ("outlier_capper", outlier_capper),
        ("feature_engineer", FeatureEngineer()),
        ("cat_encoder", cat_encoder),
        ("scaler", scaler),
        ("feature_dropper", feat_dropper),
        ("oversampler", oversampler),
        ("dtype_converter", dtype_converter)
    ])

    return final_pipeline