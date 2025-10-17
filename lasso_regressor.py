from sklearn.linear_model import Lasso
import data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt


def run_lasso(data_preprocessing):
    """Запуск Lasso Regressor"""
    print("___ Lasso Regressor ___")
    
    X_train, X_test, y_train, y_test, feature_names, _, _ = data_preprocessing
    
    # Создание и обучение модели
    lasso_model = Lasso(
        alpha=0.01,  # параметр регуляризации
        max_iter=10000,
        random_state=42,
        selection='random'
    )
    
    lasso_model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = lasso_model.predict(X_test)
    
    # Оценка модели
    metrics = dp.evaluate_model(y_test, y_pred, "Lasso Regressor")
    
    # Визуализация результатов
    dp.plot_results(y_test, y_pred, "Lasso Regressor")
    
    # Коэффициенты модели
    coefficients = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lasso_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    

    
    # Ненулевые коэффициенты
    non_zero_coeffs = coefficients[coefficients['coefficient'] != 0]
    
    # График коэффициентов
    plt.figure(figsize=(10, 6))
    top_coeffs = coefficients.head(10).sort_values('coefficient', ascending=True)
    plt.barh(top_coeffs['feature'], top_coeffs['coefficient'])
    plt.title('Lasso Regressor: Коэффициенты признаков (Топ-10)')
    plt.xlabel('Значение коэффициента')
    plt.tight_layout()
    plt.show()
    
    return lasso_model, metrics


if __name__ == "__main__":
    df = dp.load_data()
    data_preprocessing = dp.preprocess_data(df)
    run_lasso(data_preprocessing)