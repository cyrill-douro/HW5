import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_preprocessing as dp
from extra_trees_regressor import run_extra_trees
from lgbm_regressor import run_lgbm
from random_forest_regressor import run_random_forest
from huber_regressor import run_huber
from lasso_regressor import run_lasso


def compare_models(data_preprocessing):
    """Сравнение всех регрессоров"""
    print("___ СРАВНЕНИЕ РЕГРЕССОРОВ ___")
    
    # Запуск всех моделей
    models_metrics = {}
    
    
    _, et_metrics = run_extra_trees(data_preprocessing)
    models_metrics['Extra Trees'] = et_metrics
    
   
    _, lgbm_metrics = run_lgbm(data_preprocessing)
    models_metrics['LightGBM'] = lgbm_metrics
    
   
    _, rf_metrics = run_random_forest(data_preprocessing)
    models_metrics['Random Forest'] = rf_metrics
    
    
    _, huber_metrics = run_huber(data_preprocessing)
    models_metrics['Huber'] = huber_metrics
    
   
    _, lasso_metrics = run_lasso(data_preprocessing)
    models_metrics['Lasso'] = lasso_metrics
    
    # Создание DataFrame для сравнения
    comparison_df = pd.DataFrame(models_metrics).T
    
    # Вывод результатов сравнения
    print("\n___ ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ ___")
    print(comparison_df.round(4))
    
    # Визуализация сравнения
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График MSE
    comparison_df['MSE'].plot(kind='bar', ax=axes[0,0], color='skyblue', alpha=0.7)
    axes[0,0].set_title('Сравнение MSE')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # График MAE
    comparison_df['MAE'].plot(kind='bar', ax=axes[0,1], color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Сравнение MAE')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # График RMSE
    comparison_df['RMSE'].plot(kind='bar', ax=axes[1,0], color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Сравнение RMSE')
    axes[1,0].set_ylabel('RMSE')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # График R²
    comparison_df['R2'].plot(kind='bar', ax=axes[1,1], color='gold', alpha=0.7)
    axes[1,1].set_title('Сравнение R²')
    axes[1,1].set_ylabel('R² Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Тепловая карта метрик
    plt.figure(figsize=(10, 6))
    sns.heatmap(comparison_df, annot=True, cmap='YlOrRd', fmt='.4f', linewidths=0.5)
    plt.title('Тепловая карта метрик регрессоров')
    plt.tight_layout()
    plt.show()
    
    # Определение лучшей модели по R²
    best_model_r2 = comparison_df['R2'].idxmax()
    best_r2_score = comparison_df['R2'].max()
    
    # Определение лучшей модели по MSE
    best_model_mse = comparison_df['MSE'].idxmin()
    best_mse_score = comparison_df['MSE'].min()
    
    print(f"\n___ ЛУЧШИЕ МОДЕЛИ ___")
    print(f"Лучшая модель по R²: {best_model_r2} (R² = {best_r2_score:.4f})")
    print(f"Лучшая модель по MSE: {best_model_mse} (MSE = {best_mse_score:.4f})")
    
    return comparison_df


if __name__ == "__main__":
    df = dp.load_data()
    data_preprocessing = dp.preprocess_data(df)
    results = compare_models(data_preprocessing)