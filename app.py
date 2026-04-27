from PySide6 import QtWidgets
from PySide6 import QtGui
from PySide6 import QtCore
import RAI
import os
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.wayland.textinput=false'

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rubber End AI :)')
        self.resize(800, 600)
        self.__selected_mode = None
        self.show_mode_selection()

    def show_mode_selection(self):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setSpacing(15)
        layout.addStretch(1)
        title = QtWidgets.QLabel('RAI Select Mode:')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        for mode in ['default', 'train']:
            btn = QtWidgets.QPushButton(mode.capitalize())
            btn.setFixedHeight(200)
            btn.setFixedWidth(200)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(lambda chk = False, m = mode: self.set_app_mode(m))
            layout.addWidget(btn, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch(1)
        self.setCentralWidget(container)

    def set_app_mode(self, mode):
        self.__selected_mode = mode
        self.start_default_flow() if mode == 'default' else self.start_train_flow()

    def start_default_flow(self):
        pth_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'RAI: Checkpoint file', 'checkpoint.pth', 'Model files (*.pth);;All Files (*)'
        )
        if not pth_path: return

        password, ok = QtWidgets.QInputDialog.getText(
            self, 'RAI: Password to Checkpoint file', 'Enter the password to Сheckpoint:', QtWidgets.QLineEdit.Password
        )
        if not ok or not password: return

        input_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'RAI: Input file', 'input.yaml', 'YAML files (*.yaml);;All Files (*)'
        )
        if not input_path: return

        self.show_yaml_editor(input_path, pth_path, password)

    def start_train_flow(self):
        ds_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'RAI: Dataset file', 'dataset.csv', 'CSV Files (*.csv);;All Files (*)'
        )
        if not ds_path: return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'RAI: Checkpoint file', 'checkpoint.pth', 'Model files (*.pth);;All Files (*)'
        )
        if not save_path: return

        epochs, ok1 = QtWidgets.QInputDialog.getInt(
            self, 'RAI: Epochs', 'Epochs:', 100, 1, 1000000
        )
        if not ok1: return

        patience, ok2 = QtWidgets.QInputDialog.getInt(
            self, 'RAI: Patience', 'Patience:', 10, 1, 500
        )
        if not ok2: return


        while True:
            lr_text, ok3 = QtWidgets.QInputDialog.getText(
                self, 'RAI: Learning Rate', ' Learning Rate(0.0000001 < lr < 1):',
                text = '0.0001'
            )
            if not ok3: return

            try:
                lr = float(lr_text.replace(',', '.'))
                if lr > 0.0000001 and lr < 1: break
            except ValueError: pass

        self.setup_train_ui(ds_path, save_path, epochs, patience, lr)

    def setup_train_ui(self, ds_path, save_path, epochs, patience, lr):
        try:
            import torch
            container = QtWidgets.QWidget()

            main_layout = QtWidgets.QHBoxLayout(container)
            main_layout.setSpacing(5)

            left_panel = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)

            self.trainer_instance = RAI.Trainer()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            title = QtWidgets.QLabel('Info:')
            left_layout.addWidget(title)

            params = [
                f'Device: {device}',
                f'Epochs: {epochs}',
                f'Patience: {patience}',
                f'Learning Rate: {lr}'
            ]

            for param in params:
                lbl = QtWidgets.QLabel(param)
                left_layout.addWidget(lbl)

            left_layout.addStretch(1)

            buttons_row = QtWidgets.QVBoxLayout()

            menu_btn = QtWidgets.QPushButton('Main Menu')
            menu_btn.setFixedHeight(100)
            menu_btn.setFixedWidth(200)

            start_btn = QtWidgets.QPushButton('START')
            start_btn.setFixedHeight(100)
            start_btn.setFixedWidth(200)

            stop_btn = QtWidgets.QPushButton('STOP')
            stop_btn.setFixedHeight(100)
            stop_btn.setFixedWidth(200)

            stop_btn.setEnabled(False)
            start_btn.clicked.connect(lambda: menu_btn.setEnabled(False))
            start_btn.clicked.connect(lambda: start_btn.setEnabled(False))
            start_btn.clicked.connect(lambda: stop_btn.setEnabled(True))
            start_btn.clicked.connect(lambda: self.run_training(device, ds_path, epochs, patience, lr, save_path))
            stop_btn.clicked.connect(self.trainer_instance.stop)
            stop_btn.clicked.connect(lambda: menu_btn.setEnabled(True))
            stop_btn.clicked.connect(lambda: start_btn.setEnabled(True))
            stop_btn.clicked.connect(lambda: stop_btn.setEnabled(False))

            buttons_row.addWidget(menu_btn)
            buttons_row.addWidget(start_btn)
            buttons_row.addWidget(stop_btn)
            left_layout.addLayout(buttons_row)
            left_layout.addStretch(1)

            right_panel = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right_panel)

            console_label = QtWidgets.QLabel('Console Output:')
            right_layout.addWidget(console_label)

            self.log_console = QtWidgets.QTextEdit()
            self.log_console.setReadOnly(True)
            right_layout.addWidget(self.log_console)

            main_layout.addWidget(left_panel, stretch=1)
            main_layout.addWidget(right_panel, stretch=3)

            self.setCentralWidget(container)
            self.setWindowTitle('RAI: Training')
            menu_btn.clicked.connect(self.go_to_main_menu)


        except Exception as e:
            QtWidgets.QMessageBox.critical(self, '[ FAILED ]: Init', str(e))


    def request_stop(self):
        if hasattr(self, 'trainer'):
            self.trainer.stop_signal = True
            self.log_console.append('\n[ WARN ]: Stop...')

    def run_training(self, device, ds_path, epochs, patience, lr, save_path):
        dataset = RAI.Loader.load_dataset(ds_path)
        self.log_console.append(f'[ SUCCESS ]: Dataset loaded: {len(dataset)} rows')
        trainer = self.trainer_instance

        def update_logs(logs=None):
            if logs is not None:
                cnt = logs.get('counter', 0)
                patience_info = f' | StopCounter: {cnt}' if cnt > 0 else ''

                msg = f'MSE_km: {logs['MSE_km']} | MSE: {logs['MSE']}{patience_info}'
                self.log_console.append(msg)

            QtWidgets.QApplication.processEvents()

        trainer.run(
            device = device,
            dataset = dataset,
            epochs = epochs,
            patience = patience,
            learning_rate = lr,
            log_callback = update_logs
        )

        if hasattr(trainer, '_Trainer__is_running') and not trainer._Trainer__is_running:
            self.log_console.append('\n[ WARN ]: Training was interrupted manually.')

        password, ok = QtWidgets.QInputDialog.getText(self, 'Save', 'Set password for encryption:', QtWidgets.QLineEdit.Password)
        if ok:
            key = RAI.KeyGen(password).get()
            RAI.Checkpoint.save(trainer.get(), key, save_path)
            self.log_console.append(f'\n[ SUCCESS ]: Saved to {save_path}')

    def show_yaml_editor(self, file_path, pth_path=None, password=None):
        import os
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        self.text_edit = QtWidgets.QTextEdit()

        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.text_edit.setText(f.read())

        layout.addWidget(self.text_edit)

        menu_btn = QtWidgets.QPushButton('Main Menu')
        run_btn = QtWidgets.QPushButton('Get a prediction')
        run_btn.clicked.connect(lambda: self.on_predict_clicked(pth_path, password))
        layout.addWidget(menu_btn)
        layout.addWidget(run_btn)

        self.setCentralWidget(container)
        menu_btn.clicked.connect(self.go_to_main_menu)

    def on_predict_clicked(self, pth_path, password):
        import yaml

        try:
            key = RAI.KeyGen(password).get()
            checkpoint = RAI.Checkpoint.load(pth_path, key, 'cpu')

            yaml_text = self.text_edit.toPlainText()
            user_input_data = yaml.safe_load(yaml_text)

            predictor = RAI.Predictor(checkpoint, user_input_data)
            result = predictor.get()

            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle('RAI: Result')
            msg_box.setText(f'Result: {result} km')
            msg_box.setInformativeText('Do you want to save this result to a file?')
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Close)
            msg_box.setDefaultButton(QtWidgets.QMessageBox.Save)

            if msg_box.exec() == QtWidgets.QMessageBox.Save:
                save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'RAI: Save Result', 'result.txt', 'Text Files (*.txt);;All Files (*)'
                )
                if save_path:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(f'--- Rubber End AI Report ---\n')
                        f.write(f'Result: {result} km\n')
                        f.write(f'Input configuration used:\n')
                        f.write(self.text_edit.toPlainText())
                    self.statusBar().showMessage(f'The result was saved to {save_path}')

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                'RAI: Error',
                'Oops! Something\'s wrong!\nPlease check if you entered the information correctly.\n'
                f'This might be a program error.\nError:\n\n{str(e)}'
            )

    def go_to_main_menu(self):
        self.show_mode_selection()
