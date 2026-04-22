# Rubber End AI - специализированная нейросетевая модель для расчёта срока службы машинных шин.

Цель проекта - определять точное время, когда шины приходят в негодность, чтобы снизить расходы на создание новых шин и улучшить экологию.
Этот проект актуален потому, что изношенные шины - это большая проблема для экологии, из-за проблем с переработкой. Программа не имеет GUI и предназначена для пользователей, которые владеют базовыми навыками программирования и ручной настройки через config.

## Использование
  1. Запустите программу.<br>
  2. Придумайте свой логин и пароль.<br>
  3. Перейдите в вашу директорию (её имя будет выведено после первого ввода логина и пароля).<br>
  4. Откройте файл config.yaml и настройте его.<br>
  5. Повторите запуск и ввод логина и пароля для получения другого результата.

## Сборка
### Windows
1. Установите Python 3 (желательно 3.13 или новее) с официального сайта https://python.org или соберите вручную.<br>
2. Установите git с официального сайта https://git-scm.com.<br>
"Вы можете использовать Scoop для удобной установки: https://scoop.sh"<br>
3. Откройте окно Powershell, клонируйте репозиторий, создайте venv, установите зависимости:
```powershell
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
python -m venv .venv
.venv\Scripts\activate.ps1
pip install torch pandas pyYAML scikit-learn cryptography
```

### Большинство Linux дистрибутивов
"Рекомендуется использовать оболчку bash или zsh, но можете использовать и fish или ksh"<br>
Установите Python 3 (желательно 3.13 или новее) через пакетный менеджер вашей системы, с официального сайта https://python.org или соберите вручную.<br>
Пример (не забудьте запускать установку с правами super user):<br>
Arch\EndeavourOS\CachyOS\Manjaro
```bash
pacman -S git python python-pip
```

Ubuntu\Debian\RPIOS\LinuxMint
```bash
apt install git python3 python3-pip
```

Fedora\CENTOS\RHEL
```bash
dnf install git python3 python3-pip
```

```bash
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
python -m venv .venv
source .venv/bin/activate
pip install torch pandas pyYAML scikit-learn cryptography
```

### MacOS
"Рекомендуется использовать оболчку bash или zsh, но можете использовать и fish или ksh"<br>
1. Установите Brew с официального сайта https://brew.sh.<br>
2. Установите git и python:
```bash
brew install python git
```

3. Клонируйте репозиторий с проектом, создайте venv и установите зависимости:
```bash
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
python -m venv .venv
source .venv/bin/activate
pip install torch pandas pyYAML scikit-learn cryptography
```

### NixOS
"Рекомендуется использовать оболчку bash или zsh, но можете использовать и fish или ksh"<br>
1. Установите git и клонируйте репозиторий:
```bash
nix-shell -p git
exit
git clone https://github.com/LeonidMehonoshin/rubber_end_ai.git
cd ./rubber_end_ai/
```
2. Войдите в nix-shell с необходимыми установленными зависимостями (они прописаны в файле shell.nix):
```bash
nix-shell
```
