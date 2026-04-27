import sys
from PySide6 import QtWidgets
from app import App

def main():
    qt_app = QtWidgets.QApplication(sys.argv)

    window = App()
    window.show()

    sys.exit(qt_app.exec())

if __name__ == "__main__":
    main()
