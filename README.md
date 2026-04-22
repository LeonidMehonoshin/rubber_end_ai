# Rubber End AI
## Что это:
Rubber End AI - это специализированная нейросетевая модель для расчёта срока службы машинных шин.
Цель проекта - определять точное время, когда шины приходят в негодность, чтобы снизить расходы на создание новых шин и улучшить экологию. Изношенные шины - это актуальная проблема для экологии, из-за проблем с переработкой. Программа не имеет GUI и предназначена для пользователей, которые владеют базовыми навыками программирования и ручной настройки через config.

## Использование
1. Получите файл checkpoint.pth и secret.key к нему для того, чтобы сразу начать работу.<br>
2. Запустите программу.<br>
3. Придумайте свой логин и пароль.<br>
4. Перейдите в вашу директорию (её имя будет выведено после первого ввода логина и пароля).<br>
5. Откройте файл config.yaml и настройте его.<br>
6. Повторите запуск и ввод логина и пароля для получения другого результата.
Чтобы настроить вывод зайдите в файл inputs.yaml.

## Сборка
### Windows
#### 1. Установите Python 3 (желательно 3.13 или новее) с официального сайта https://python.org, установите git с официального сайта https://git-scm.com (Альтернативно вы можете использовать Scoop для удобной установки: https://scoop.sh)<br>
#### 2. Откройте окно Powershell, клонируйте репозиторий, и создайте venv
```powershell
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
python -m venv .venv
.venv\Scripts\activate.ps1
```

#### 3. Установите зависимости
```powershell
pip install torch pandas pyYAML scikit-learn cryptography
```

### Большинство Linux дистрибутивов
Рекомендуется использовать оболчку bash или zsh, но можете использовать и fish или ksh<br>
#### 1. Установите Python 3 (желательно 3.13 или новее) через пакетный менеджер вашей системы, с официального сайта https://python.org или соберите вручную.<br>
Пример (не забудьте запускать установку с правами super user):<br>
Arch Linux \ Endeavour OS \ Cachy OS \ Manjaro
```bash
pacman -S git python python-pip
```

Ubuntu \ Debian \ Raspberry pi OS \ Linux Mint
```bash
apt install git python3 python3-pip
```

Fedora \ CENTOS \ RHEL
```bash
dnf install git python3 python3-pip
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
pip install torch pandas pyYAML scikit-learn cryptography
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
pip install torch pandas pyYAML scikit-learn cryptography
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
