from sklearn.ensemble import ExtraTreesRegressor
import data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_extra_trees(data_preprocessing):
    """Запуск Extra Trees Regressor"""
    print("___ Extra Trees Regressor ___")
    
    X_train, X_test, y_train, y_test, feature_names, _, _ = data_preprocessing
    
    # Создание и обучение модели
    et_model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    et_model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = et_model.predict(X_test)
    
    # Оценка модели
    metrics = dp.evaluate_model(y_test, y_pred, "Extra Trees Regressor")
    
    # Визуализация результатов
    dp.plot_results(y_test, y_pred, "Extra Trees Regressor")
    
    # Важность признаков
    feature_importance = et_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    
    # График важности признаков
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp_df.head(10), x='importance', y='feature')
    plt.title('Extra Trees: Важность признаков (Топ-10)')
    plt.tight_layout()
    plt.show()
    
    return et_model, metrics


if __name__ == "__main__":
    df = dp.load_data()
    data_preprocessing = dp.preprocess_data(df)
    run_extra_trees(data_preprocessing)