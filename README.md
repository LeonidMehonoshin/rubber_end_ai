# ПРОЕКТ НЕ ЗАВЕРШЕН!
### Rubber End AI - специализированная нейросетевая модель для расчёта срока службы машинных шин.

Цель проекта - определять точное время, когда шины приходят в негодность, чтобы снизить расходы на создание новых шин и улучшить экологию.
Этот проект актуален потому, что изношенные шины - это большая проблема для экологии, из-за проблем с переработкой.

## Как работает
  1. Поместите файл checkpoint.pth в корневой каталог.<br>
  2. Если вы хотите обучить ее на своем датасете, тогда поместите файл dataset.csv в корневой каталог.<br>
  3. Настройте работу через файл config.yaml.

## Сборка
### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate.ps1
pip install torch pandas numpy pyYAML scikit-learn
```

### Linux (Debian\RHEL\Arch) или MacOS
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pandas numpy pyYAML scikit-learn
```

### NixOS Linux (пока только вручную через nix-shell, без shell.nix или flake.nix)
```bash
nix-shell -p python3 'python3.withPackages (ps: with ps; [ torch pandas numpy pyyaml scikit-learn ])'
```
