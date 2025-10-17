import os
import pandas as pd

def load_data(file_path):
    """
    Загрузка данных из CSV файла с проверкой существования файла.
    :param file_path: Путь к CSV файлу.
    :return: DataFrame с загруженными данными.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    
    df = pd.read_csv(file_path)
    print(f"Данные успешно загружены. Размер: {df.shape}")
    return df