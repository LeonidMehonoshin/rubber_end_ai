# Инструкция по установке Rubber End AI
Для работы требуется Python 3.13+ и виртуальное окружение (для установки библиотек).
---

## 1. Проверка Python
Убедитесь, что установлен [Python](https://www.python.org/downloads/) 3.13 или новее:
```bash
python --version
```

## 2. Создание окружения
Перейдите в папку проекта и создайте виртуальное окружение venv:
```bash
cd Rubber_End_AI
python -m venv venv
```

## 3. Активация окружения
Активируйте окружение для вашей ОС:
* Linux / macOS (Terminal):
  ```bash
  source venv/bin/activate
  ```
* Windows (CMD):
  ```cmd
  venv\Scripts\activate.bat
  ```
* Windows (PowerShell):
  ```powershell
  venv\Scripts\Activate.ps1
  ```

## 4. Установка зависимостей
Установите библиотеки:
```bash
pip install torch pandas pyYAML scikit-learn PySide6
```

## 5. Проверка
Проверьте корректность установки:
```bash
python -c "import torch, pandas, yaml, sklearn, PySide6; print('Установка завершена успешно')"
```

## 6. Запуск
```bash
python main.py
```
