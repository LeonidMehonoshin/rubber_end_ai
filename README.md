# ПРОЕКТ НЕ ЗАВЕРШЕН!
### Rubber End AI - специализированная нейросетевая модель для расчёта срока службы машинных шин.

Цель проекта - определять точное время, когда шины приходят в негодность, чтобы снизить расходы на создание новых шин и улучшить экологию.
Этот проект актуален потому, что изношенные шины - это большая проблема для экологии, из-за проблем с переработкой.

## Использование
  1. Запустите программу.<br>
  2. Придумайте свой логин и пароль.<br>
  3. Перейдите в вашу директорию (её имя будет выведено после первого ввода логина и пароля).<br>
  4. Откройте файл config.yaml и настройте его.<br>
  5. Повторите запуск и ввод логина и пароля для получения другого результата.

## Сборка
### Windows
#### Установите Python 3.13 с официального сайта https://python.org или соберите вручную.
```powershell
python -m venv .venv
.venv\Scripts\activate.ps1
pip install torch pandas pyYAML scikit-learn cryptography
```

### Linux (Debian\RHEL\Arch\Alt\Alpine и их производные) или MacOS
#### Установите Python 3.13 через пакетный менеджер, с официального сайта https://python.org или соберите вручную.
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pandas pyYAML scikit-learn cryptography
```

### NixOS Linux
```bash
cd "path/to/rubber_end_ai"
nix-shell
```
