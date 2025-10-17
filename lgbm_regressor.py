import lightgbm as lgb
import data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_lgbm(data_preprocessing):
    """Запуск LightGBM Regressor"""
    print("___ LightGBM Regressor ___")
    
    X_train, X_test, y_train, y_test, feature_names, _, _ = data_preprocessing
    
    # Создание и обучение модели
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    lgb_model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = lgb_model.predict(X_test)
    
    # Оценка модели
    metrics = dp.evaluate_model(y_test, y_pred, "LightGBM Regressor")
    
    # Визуализация результатов
    dp.plot_results(y_test, y_pred, "LightGBM Regressor")
    
    # Важность признаков
    feature_importance = lgb_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    
    # График важности признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp_df.head(10), x='importance', y='feature')
    plt.title('LightGBM: Важность признаков (Топ-10)')
    plt.tight_layout()
    plt.show()
    
    return lgb_model, metrics


if __name__ == "__main__":
    df = dp.load_data()
    data_preprocessing = dp.preprocess_data(df)
    run_lgbm(data_preprocessing)