from sklearn.ensemble import RandomForestRegressor
import data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_random_forest(data_preprocessing):
    """Запуск Random Forest Regressor"""
    print("___ Random Forest Regressor ___")
    
    X_train, X_test, y_train, y_test, feature_names, _, _ = data_preprocessing
    
    # Создание и обучение модели
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = rf_model.predict(X_test)
    
    # Оценка модели
    metrics = dp.evaluate_model(y_test, y_pred, "Random Forest Regressor")
    
    # Визуализация результатов
    dp.plot_results(y_test, y_pred, "Random Forest Regressor")
    
    # Важность признаков
    feature_importance = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    

    
    # График важности признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp_df.head(10), x='importance', y='feature')
    plt.title('Random Forest: Важность признаков (Топ-10)')
    plt.tight_layout()
    plt.show()
    
    return rf_model, metrics


if __name__ == "__main__":
    df = dp.load_data()
    data_preprocessing = dp.preprocess_data(df)
    run_random_forest(data_preprocessing)