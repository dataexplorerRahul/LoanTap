from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.data_preprocessing.transformers import DataCleaner, Imputer, OutlierHandler, FeatureEngineer, FeatureDropper, CatEncoder, Scaler


# Build pipeline
def build_pipeline(
        numerical_features, 
        engineered_features,
        categorical_features,
        supervised_features,
        use_imputation=True, 
        use_outlier_capping=True, 
        use_encoding=True, 
        encoder_type="woe", 
        use_scaling=True,
        use_smote=False, 
        features_to_drop=None
    ):
    # Feature types
    ordinal_features = ["grade", "sub_grade", "emp_length"]
    ohe_features = list(set(categorical_features).difference((set(ordinal_features)).union(supervised_features)))
    ohe_features = ohe_features + ["negative_rec"] # Include binary negative_rec after feature engineering
    scaling_features = numerical_features + engineered_features + ordinal_features # including ordinal-encoded features for scaling

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
        cat_encoder = CatEncoder(encoder_type, ohe_features=ohe_features, supervised_features=supervised_features)
    else:
        cat_encoder = "passthrough"
    
    #-----Dropping specified features-----
    if features_to_drop:
        feat_dropper = FeatureDropper(features=features_to_drop)
    else:
        feat_dropper = "passthrough"
    
    #-----Scaling-----
    if use_scaling:
        scaler = Scaler(features=scaling_features)
    else:
        scaler = "passthrough"

    #-----Oversampling-----
    if use_smote:
        oversampler = SMOTE()
    else:
        oversampler = "passthrough"

    # Preprocessing pipeline
    final_pipeline = Pipeline(steps=[
        ("data_cleaner", DataCleaner()),
        ("imputer", imputer),
        ("outlier_capper", outlier_capper),
        ("feature_engineer", FeatureEngineer()),
        ("cat_encoder", cat_encoder),
        ("scaler", scaler),
        ("feature_dropper", feat_dropper),
        ("oversampler", oversampler)
    ])

    return final_pipeline