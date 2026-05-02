# Rubber End AI
## Что это:
Rubber End AI - это специализированная нейросетевая модель для расчёта срока службы машинных шин.

## Использование
#### 1. Получите файл checkpoint.pth и пороль к нему.
#### 2. Выберете режим Default, импортируйте оригинальный/сторонний checkpoint.pth, введите пороль и импортируйте оригинальный/сторонний input.yaml для вашего Checkpoint. Вам будет предложено отредактировать его настройки. Примечание: input.yaml для оригинального Checkpoint поставляется уже в комплекте с программой.
#### 3. Если вы хотите использовать свой checkpoint.pth, то вы можете сгенерировать его в режиме Train. Для этого вам понадобятся базовые знания о работе нейросетей и файл датасета.

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
