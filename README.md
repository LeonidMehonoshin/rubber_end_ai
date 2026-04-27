# Rubber End AI
## Что это:
Rubber End AI - это специализированная нейросетевая модель для расчёта срока службы машинных шин.

## Использование
#### 1. Получите файл checkpoint.pth и пороль к нему.
#### 2. Выберете режим Default, импортируйте checkpoint.pth, введите пороль и импортируйте оригинальный/сторонний input.yaml. Вам будет предложено отредактировать его настройки.
#### 3. Если вы хотите использовать свой checkpoint.pth, то вы можете сгенерировать его в режиме Train. Для этого вам понадобятся базовые знания о работе нейросетей и файл датасета.
Чтобы настроить вывод зайдите в файл input.yaml.

## Сборка
### Windows
#### 1. Установите Python 3.13 (или новее) с официального сайта https://python.org, установите git с официального сайта https://git-scm.com (Альтернативно вы можете использовать Scoop для удобной установки: https://scoop.sh)<br>
#### 2. Откройте окно Powershell, клонируйте репозиторий, и создайте venv
```powershell
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
python -m venv .venv
.venv\Scripts\activate.ps1
```

#### 3. Установите зависимости
```powershell
pip install torch pandas pyYAML scikit-learn cryptography PySide6
```

### Большинство Linux дистрибутивов
Рекомендуется использовать оболчку bash или zsh, но можете использовать и fish или ksh<br>
#### 1. Установите Python 3.13 (или новее) через пакетный менеджер вашей системы, с официального сайта https://python.org или соберите вручную.<br>
Пример (не забудьте запускать установку с правами super user):<br>
Arch Linux | Endeavour OS | Cachy OS | Manjaro
```bash
pacman -S git python
```

Ubuntu | Debian | Raspberry pi OS | Linux Mint
```bash
apt install git python3
```

Fedora | CENTOS | RHEL
```bash
dnf install git python3
```

#### 2. Клонируйте репозиторий и создайте venv
```bash
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
python -m venv .venv
source .venv/bin/activate
```

#### 3. Установите зависимости
```bash
pip install torch pandas pyYAML scikit-learn cryptography PySide6
```

### MacOS
Рекомендуется использовать оболчку bash или zsh, но можете использовать и fish или ksh<br>
#### 1. Установите Brew с официального сайта https://brew.sh.<br>
#### 2. Установите git и python
```bash
brew install python git
```

#### 3. Клонируйте репозиторий и создайте venv
```bash
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
python -m venv .venv
source .venv/bin/activate
```

#### 4. Установите зависимости
```bash
pip install torch pandas pyYAML scikit-learn cryptography PySide6
```

### NixOS
Рекомендуется использовать оболчку bash или zsh, но можете использовать и fish или ksh<br>
1. Установите git в nix-shell
```bash
nix-shell -p git
```
2. Клонируйте репозиторий
```nix-shell
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
exit
```
3. Войдите в nix-shell с необходимыми установленными зависимостями (они прописаны в файле shell.nix)
```bash
nix-shell
```

## Скриншоты
<img width="1447" height="129" alt="Screenshot from 2026-04-23 18-14-19" src="https://github.com/user-attachments/assets/3c76bd5f-ac10-4c42-8e6d-366fb36029f7" /><br>
<img width="1726" height="379" alt="Screenshot from 2026-04-23 18-41-37" src="https://github.com/user-attachments/assets/e1498598-6b0d-449e-9044-1d9e2621deaf" /><br>
<img width="1551" height="153" alt="Screenshot from 2026-04-23 18-42-45" src="https://github.com/user-attachments/assets/e6719cb6-7610-43e9-a231-3e75f8c2b6e0" /><br>
<img width="1585" height="131" alt="Screenshot from 2026-04-23 18-43-21" src="https://github.com/user-attachments/assets/970f9e51-7912-44af-adef-551667711068" />
