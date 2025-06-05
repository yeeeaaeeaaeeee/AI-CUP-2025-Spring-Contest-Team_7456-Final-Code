import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

def KFoldById(X, y, player_ids, model_setting, RANDOM_SEED, K=3, multi=False, model_type=XGBClassifier, top_n_features=None):
    """
    KFold with feature selection by importance.
    If top_n_features is a list, returns a list of masks (indices) for each value.
    """
    cv = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = []
    feature_importances = np.zeros(X.shape[1] if hasattr(X, 'shape') else len(X[0]))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=player_ids)):
        train_players = np.unique(player_ids[train_idx])
        val_players = np.unique(player_ids[val_idx])
        val_genders = y[val_idx].astype(int)
        tr_genders = y[train_idx].astype(int)
        print(f"Fold {fold}: val_players={len(val_players)}, class counts={np.bincount(val_genders), np.bincount(tr_genders)}")
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = model_type(**model_setting)
        model.fit(X_tr, y_tr)
        # Accumulate feature importances
        if hasattr(model, "feature_importances_"):
            feature_importances += model.feature_importances_
        else:
            print("Model does not have feature_importances_ attribute.")
        
        y_pred = model.predict_proba(X_val) if multi else model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred, average='micro', multi_class='ovr') if multi else roc_auc_score(y_val, y_pred, average='micro')
        auc_scores.append(auc)
    print(f"Cross-validated ROC AUC scores: {auc_scores}")
    print(f"Mean ROC AUC: {np.mean(auc_scores):.4f}")

    if top_n_features is None:
        return

    # Average feature importances
    feature_importances /= K

    # If top_n_features is a list, return a list of masks
    if isinstance(top_n_features, (list, tuple, np.ndarray)):
        masks = []
        for n in top_n_features:
            top_indices = np.argsort(feature_importances)[::-1][:n]
            masks.append(top_indices)
        return masks

    # If top_n_features is a single int, return one mask
    top_indices = np.argsort(feature_importances)[::-1][:top_n_features]
    return top_indices

def KFoldEnsembleByMasks(X, y, player_ids, model_setting, RANDOM_SEED, masks, K=3, multi=False, model_type=XGBClassifier):
    """
    KFold ensemble validation using multiple feature masks.
    Trains one model per mask, then averages predictions for validation.
    Returns average AUC, list of trained models, and out-of-fold predictions aligned with X.
    """
    cv = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = []
    all_models = [[] for _ in range(len(masks))]
    oof_preds = np.zeros((len(X), np.unique(y).size if multi else 1))  # For multiclass, shape (n_samples, n_classes)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=player_ids)):
        print(f"Fold {fold}:")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        val_preds = []
        for i, mask in enumerate(masks):
            model = model_type(**model_setting)
            model.fit(X_tr.iloc[:, mask], y_tr)
            all_models[i].append(model)
            if multi:
                pred = model.predict_proba(X_val.iloc[:, mask])
            else:
                pred = model.predict_proba(X_val.iloc[:, mask])[:, 1]
            val_preds.append(pred)
        # Average predictions across models
        avg_pred = np.mean(val_preds, axis=0)
        if multi:
            auc = roc_auc_score(y_val, avg_pred, average='micro', multi_class='ovr')
            oof_preds[val_idx, :] = avg_pred
        else:
            auc = roc_auc_score(y_val, avg_pred, average='micro')
            oof_preds[val_idx, 0] = avg_pred
        auc_scores.append(auc)
        print(f"  AUC: {auc:.4f}")

    print(f"Cross-validated ROC AUC scores: {auc_scores}")
    print(f"Mean ROC AUC: {np.mean(auc_scores):.4f}")
    # If binary, flatten oof_preds to 1D
    if not multi:
        oof_preds = oof_preds.ravel()
    return auc_scores, all_models, oof_preds

def get_test_ensemble_results(X_train, y_train, X_test, model_setting, masks, multi=False, model_type=XGBClassifier):
    """
    Train an ensemble of models using different feature masks and average their predictions.
    Args:
        X_train: Training features (DataFrame)
        y_train: Training labels
        X_test: Test features (DataFrame)
        model_setting: Model hyperparameters (dict)
        masks: List of feature index arrays (length = number of models)
        multi: If True, use multi-class probabilities
        model_type: Model class (e.g., XGBClassifier)
    Returns:
        Averaged predictions across all models for test and train.
    """
    preds_test = []
    preds_train = []
    for mask in masks:
        model = model_type(**model_setting)
        model.fit(X_train.iloc[:, mask], y_train)
        if multi:
            pred_test = model.predict_proba(X_test.iloc[:, mask])
            pred_train = model.predict_proba(X_train.iloc[:, mask])
        else:
            pred_test = model.predict_proba(X_test.iloc[:, mask])[:, 1]
            pred_train = model.predict_proba(X_train.iloc[:, mask])[:, 1]
        preds_test.append(pred_test)
        preds_train.append(pred_train)
    # Average predictions across all models
    avg_pred_test = np.mean(preds_test, axis=0)
    avg_pred_train = np.mean(preds_train, axis=0)
    return avg_pred_test, avg_pred_train