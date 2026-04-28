from PySide6 import QtWidgets
from PySide6 import QtCore
import RAI

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rubber End AI :)')
        self.resize(800, 600)
        self.__show_mode_selection()

    def __show_mode_selection(self):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setSpacing(15)
        layout.addStretch(1)
        title = QtWidgets.QLabel('RAI Select Mode:')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        for mode in ['default', 'train']:
            button = QtWidgets.QPushButton(mode.capitalize())
            button.setFixedHeight(200)
            button.setFixedWidth(200)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.clicked.connect(self.__default_mode_window if mode == 'default' else self.__setup_train_ui)
            layout.addWidget(button, alignment = QtCore.Qt.AlignCenter)

        layout.addStretch(1)
        self.setCentralWidget(container)

    def __setup_train_ui(self):
        self.__log_console = QtWidgets.QTextEdit()
        self.__log_console.setReadOnly(True)

        paths = {}
        paths['dataset'], _ = QtWidgets.QFileDialog.getOpenFileName(self, 'RAI: Dataset file', 'dataset.csv', 'CSV Files (*.csv);;All Files (*)')
        if not paths['dataset']: return
        paths['checkpoint'], _ = QtWidgets.QFileDialog.getSaveFileName(self, 'RAI: Checkpoint file', 'checkpoint.pth', 'Model files (*.pth);;All Files (*)')
        if not paths['checkpoint']: return

        options = {}
        options['Epochs'], ok = QtWidgets.QInputDialog.getInt(self, 'RAI: Epochs', 'Epochs:', 100, 1, 1000000)
        if not ok: return
        options['Patience'], ok = QtWidgets.QInputDialog.getInt(self, 'RAI: Patience', 'Patience:', 10, 1, 500)
        if not ok: return

        while True:
            lr_text, ok = QtWidgets.QInputDialog.getText(
                self, 'RAI: Learning Rate', ' Learning Rate(0.0000001 < lr < 1):',
                text = '0.0001'
            )
            if not ok: return

            try:
                options['Learning Rate'] = float(lr_text.replace(',', '.'))
                if options['Learning Rate'] > 0.0000001 and options['Learning Rate'] < 1: break
            except ValueError: pass

        try:
            import torch
            containers = {}
            for name in ('main', 'left', 'right'): containers[name] = QtWidgets.QWidget()

            layouts = {
                'main': QtWidgets.QHBoxLayout(containers['main']),
                'left': QtWidgets.QVBoxLayout(containers['left']),
                'right': QtWidgets.QVBoxLayout(containers['right'])
            }

            layouts['main'].setSpacing(5)
            layouts['left'].setContentsMargins(0, 0, 0, 0)

            for name, stretch in zip(('left', 'right'), (1, 3)):
                layouts['main'].addWidget(containers[name], stretch = stretch)

            layouts['left'].addWidget(QtWidgets.QLabel('Info:'))

            self.__trainer_instance = RAI.Trainer()
            options['Device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

            for key, value in options.items():
                layouts['left'].addWidget(QtWidgets.QLabel(f'{key}: {value}'))

            layouts['left'].addStretch(1)

            buttons = {
                'menu': QtWidgets.QPushButton('Main Menu'),
                'start': QtWidgets.QPushButton('START'),
                'stop': QtWidgets.QPushButton('STOP')
            }

            for button in buttons.values():
                button.setFixedHeight(100)
                button.setFixedWidth(200)
                button.setCursor(QtCore.Qt.PointingHandCursor)

            buttons['stop'].setEnabled(False)

            for button, callbacks in (
                (buttons['start'], (
                    lambda: buttons['menu'].setEnabled(False),
                    lambda: buttons['start'].setEnabled(False),
                    lambda: buttons['stop'].setEnabled(True),
                    lambda: self.__run_training(options, paths)
                )),
                (buttons['stop'], (
                    lambda: buttons['menu'].setEnabled(True),
                    lambda: buttons['start'].setEnabled(True),
                    lambda: buttons['stop'].setEnabled(False),
                    self.__trainer_instance.stop,
                    lambda: self.__log_console.append('\n[ WARN ]: Stop...')
                )),
                (buttons['menu'], (self.__show_mode_selection,))
            ):
                if not isinstance(callbacks, (list, tuple)): callbacks = [callbacks]
                for callback in callbacks:  button.clicked.connect(callback)
                layouts['left'].addWidget(button)

            layouts['left'].addStretch(1)

            layouts['right'].addWidget(QtWidgets.QLabel('Console Output:'))
            layouts['right'].addWidget(self.__log_console)

            self.setCentralWidget(containers['main'])
            self.setWindowTitle('RAI: Training')

        except Exception as err:
            QtWidgets.QMessageBox.critical(self, '[ FAILED ]: Init', f'{type(err).__name__}: {str(err)}')

    __training_log_signal = QtCore.Signal(str)
    __training_finished_signal = QtCore.Signal(dict)

    def __start_heavy_training(self, options, paths, dataset):
        self.__training_log_signal.emit(f'[ SUCCESS ]: Dataset loaded: {len(dataset)} rows')

        def update_logs(logs = None):
            if logs is not None:
                counter = logs.get('counter', 0)
                patience_info = f' | StopCounter: {counter}' if counter > 0 else ''

                msg = f'Epoch: {logs['epoch']} | MSE_km: {logs['MSE_km']} | MSE: {logs['MSE']}{patience_info}'
                self.__training_log_signal.emit(msg)

        self.__trainer_instance.run(
            device = options['Device'],
            dataset = dataset,
            epochs = options['Epochs'],
            patience = options['Patience'],
            learning_rate = options['Learning Rate'],
            log_callback = update_logs
        )

        self.__training_log_signal.disconnect()
        self.__training_finished_signal.emit(paths)

    def __run_training(self, options, paths):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        dataset = RAI.Loader.load_dataset(paths['dataset'])
        QtWidgets.QApplication.restoreOverrideCursor()

        self.__training_log_signal.connect(self.__log_console.append)
        self.__training_finished_signal.connect(self.__finalize_training)

        import threading
        thread = threading.Thread(target=self.__start_heavy_training, args=(options, paths, dataset), daemon=True)
        thread.start()

    def __finalize_training(self, paths):
        self.__training_finished_signal.disconnect()
        if self.__trainer_instance._Trainer__is_running:
            self.__log_console.append('\n[ WARN ]: Training has been stopped.')

        password, ok = QtWidgets.QInputDialog.getText(self, 'Save', 'Set password:', QtWidgets.QLineEdit.Password)
        if ok:
            key = RAI.KeyGen(password).get()
            RAI.Checkpoint.save(self.__trainer_instance.get(), key, paths['checkpoint'])
            self.__log_console.append(f'\n[ SUCCESS ]: Saved to {paths["checkpoint"]}')

    def __default_mode_window(self):
        password, ok = QtWidgets.QInputDialog.getText(self, 'RAI: Password', 'Enter password:', QtWidgets.QLineEdit.Password)
        if not ok or not password: return

        file_configs = [
            ('checkpoint', 'RAI: Checkpoint file', 'checkpoint.pth', 'Model files (*.pth);;All Files (*)'),
            ('input', 'RAI: Input file', 'input.yaml', 'YAML files (*.yaml);;All Files (*)')
        ]

        paths = {}
        for key, title, default, fmt in file_configs:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, default, fmt)
            if not path: return
            paths[key] = path

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        self.__input_text = QtWidgets.QTextEdit()

        import os
        if os.path.exists(paths['input']):
            with open(paths['input'], 'r', encoding='utf-8') as f:
                self.__input_text.setText(f.read())

        layout.addWidget(self.__input_text)

        buttons = {
            'menu': QtWidgets.QPushButton('Main Menu'),
            'run': QtWidgets.QPushButton('Get a prediction')
        }

        actions = (
            self.__show_mode_selection,
            lambda: self.__on_predict_clicked(paths['checkpoint'], password)
        )

        for button, action in zip(buttons.values(), actions):
            button.clicked.connect(action)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            layout.addWidget(button)

        self.setCentralWidget(container)

    def __on_predict_clicked(self, checkpoint_path, password):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        import yaml

        try:
            key = RAI.KeyGen(password).get()
            checkpoint = RAI.Checkpoint.load(checkpoint_path, key, 'cpu')

            yaml_text = self.__input_text.toPlainText()
            user_input_data = yaml.safe_load(yaml_text)

            predictor = RAI.Predictor(checkpoint, user_input_data)
            result = predictor.get()
            QtWidgets.QApplication.restoreOverrideCursor()

            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle('RAI: Result')
            msg_box.setText(f'Result: {result} km')
            msg_box.setInformativeText('Do you want to save this result to a file?')
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Close)
            msg_box.setDefaultButton(QtWidgets.QMessageBox.Save)

            if msg_box.exec() == QtWidgets.QMessageBox.Save:
                report_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'RAI: Save Result', 'result.txt', 'Text Files (*.txt);;All Files (*)'
                )

                if report_path:
                    with open(report_path, 'w', encoding = 'utf-8') as report_file:
                        report_file.write(f'--- Rubber End AI Report ---\n')
                        report_file.write(f'Result: {result} km\n')
                        report_file.write(f'Input configuration used:\n')
                        report_file.write(self.__input_text.toPlainText())

                    self.statusBar().showMessage(f'The result was saved to {report_path}')

        except Exception as err:
            QtWidgets.QMessageBox.critical(
                self,
                'RAI: Error',
                'Oops! Something\'s wrong!\nPlease check if you entered the information correctly.\n'
                f'This might be a program error.\nError:\n\n{str(err)}'
            )
