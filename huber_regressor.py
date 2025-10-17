from sklearn.linear_model import HuberRegressor
import data_preprocessing as dp
import pandas as pd
import matplotlib.pyplot as plt


def run_huber(data_preprocessing):
    """Запуск Huber Regressor"""
    print("____ Huber Regressor ___")
    
    X_train, X_test, y_train, y_test, feature_names, _, _ = data_preprocessing
    
    # Создание и обучение модели
    huber_model = HuberRegressor(
        epsilon=1.35,  # параметр для устойчивости к выбросам
        max_iter=1000,
        alpha=0.0001
    )
    
    huber_model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = huber_model.predict(X_test)
    
    # Оценка модели
    metrics = dp.evaluate_model(y_test, y_pred, "Huber Regressor")
    
    # Визуализация результатов
    dp.plot_results(y_test, y_pred, "Huber Regressor")
    
    # Коэффициенты модели
    coefficients = pd.DataFrame({
        'feature': feature_names,
        'coefficient': huber_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    
    # График коэффициентов
    plt.figure(figsize=(10, 6))
    top_coeffs = coefficients.head(10).sort_values('coefficient', ascending=True)
    plt.barh(top_coeffs['feature'], top_coeffs['coefficient'])
    plt.title('Huber Regressor: Коэффициенты признаков (Топ-10)')
    plt.xlabel('Значение коэффициента')
    plt.tight_layout()
    plt.show()
    
    return huber_model, metrics


if __name__ == "__main__":
    df = dp.load_data()
    data_preprocessing = dp.preprocess_data(df)
    run_huber(data_preprocessing)