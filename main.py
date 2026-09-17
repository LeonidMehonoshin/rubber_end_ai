import json
import os
import sys
import threading
import torch
import yaml
from PySide6.QtCore import QObject, Property, Signal, Slot, QUrl
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication
import RAIcore as RAI

os.environ["QT_QUICK_CONTROLS_STYLE"] = "Material"


class RAIController(QObject):
    logReceived = Signal(str)
    trainingFinished = Signal()
    isTrainingChanged = Signal()
    isCustomColorsChanged = Signal()
    isLightModeChanged = Signal()

    def __init__(self):
        super().__init__()
        self._is_training = False
        self._trainer_instance = None
        self._current_paths = {}
        self._options = {}

        self._config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self._is_custom_colors = False
        self._is_light_mode = True
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self._config_path):
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._is_custom_colors = bool(data.get("isCustomColors", False))
                self._is_light_mode = bool(data.get("isLightMode", True))
        except Exception:
            pass

    def _save_config(self):
        try:
            data = {
                "isCustomColors": self._is_custom_colors,
                "isLightMode": self._is_light_mode,
            }
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception:
            pass

    @Property(bool, notify=isCustomColorsChanged)
    def isCustomColors(self):
        return self._is_custom_colors

    @isCustomColors.setter
    def isCustomColors(self, val):
        if self._is_custom_colors != val:
            self._is_custom_colors = val
            self.isCustomColorsChanged.emit()
            self._save_config()

    @Property(bool, notify=isLightModeChanged)
    def isLightMode(self):
        return self._is_light_mode

    @isLightMode.setter
    def isLightMode(self, val):
        if self._is_light_mode != val:
            self._is_light_mode = val
            self.isLightModeChanged.emit()
            self._save_config()

    @Property(bool, notify=isTrainingChanged)
    def isTraining(self):
        return self._is_training


    def _to_local_path(self, url_str):
        if not url_str:
            return ""
        return (
            QUrl(url_str).toLocalFile()
            if url_str.startswith("file:")
            else url_str
        )

    @Slot(result=str)
    def getProjectPath(self):
        return QUrl.fromLocalFile(os.path.dirname(__file__)).toString()

    @Slot(result=str)
    def getDevice(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @Slot(str, str, int, int, float)
    def startTraining(self, dataset_url, checkpoint_url, epochs, patience, lr):
        if self._is_training:
            return

        self._current_paths = {
            "dataset": self._to_local_path(dataset_url),
            "checkpoint": self._to_local_path(checkpoint_url),
        }
        self._options = {
            "Epochs": epochs,
            "Patience": patience,
            "Learning Rate": lr,
            "Device": self.getDevice(),
        }

        self._is_training = True
        self.isTrainingChanged.emit()
        threading.Thread(target=self._heavy_training_worker, daemon=True).start()

    def _heavy_training_worker(self):
        try:
            dataset = RAI.Loader.load_dataset(self._current_paths["dataset"])
            self.logReceived.emit(f"[ SUCCESS ]: Dataset loaded: {len(dataset)} rows")

            self._trainer_instance = RAI.Trainer()

            def update_logs(logs=None):
                if logs:
                    counter = logs.get("counter", 0)
                    patience_info = (
                        f" | Patience: {counter}/{self._options['Patience']}"
                        if counter > 0
                        else " | Patience: OK"
                    )
                    self.logReceived.emit(
                        f"Epoch: {logs['epoch']} | MSE: {logs['MSE']:.4%}{patience_info}"
                    )

            self._trainer_instance.run(
                device=self._options["Device"],
                dataset=dataset,
                epochs=self._options["Epochs"],
                patience=self._options["Patience"],
                learning_rate=self._options["Learning Rate"],
                log_callback=update_logs,
            )
        except Exception as err:
            self.logReceived.emit(f"[ FAILED ]: Error during training: {err}")
        finally:
            self._is_training = False
            self.isTrainingChanged.emit()
            self.trainingFinished.emit()

    @Slot()
    def stopTraining(self):
        if self._trainer_instance and self._is_training:
            self.logReceived.emit("\n[ WAIT ]: Stopping thread...")
            self._trainer_instance.stop()


    @Slot(str)
    def saveCheckpoint(self, checkpoint_url):
        try:
            if self._trainer_instance:
                checkpoint_path = self._to_local_path(checkpoint_url)
                RAI.Checkpoint.save(self._trainer_instance.get(), checkpoint_path)
                self.logReceived.emit(f"\n[ SUCCESS ]: Saved to {checkpoint_path}")
        except Exception as err:
            self.logReceived.emit(f"[ ERROR ]: Save failed: {err}")

    @Slot(str, result=str)
    def loadYamlFile(self, yaml_url):
        try:
            with open(
                self._to_local_path(yaml_url), "r", encoding="utf-8"
            ) as f:
                return f.read()
        except Exception as err:
            return f"Error loading file: {err}"

    @Slot(str, result=str)
    def parseYamlToNative(self, yaml_text):
        if not yaml_text or yaml_text.startswith("Error loading file:"):
            return "INVALID_FORMAT"
        try:
            data = yaml.safe_load(yaml_text)
            if isinstance(data, dict) and data:
                return json.dumps(data)
        except Exception:
            pass
        return "INVALID_FORMAT"

    @Slot(str, str, result=str)
    def getPredictionFromFields(self, checkpoint_url, json_fields_text):
        try:
            checkpoint = RAI.Checkpoint.load(
                self._to_local_path(checkpoint_url), "cpu"
            )
            user_input_data = json.loads(json_fields_text)
            predictor = RAI.Predictor(checkpoint, user_input_data)
            return f"Result: {predictor.get()} km"
        except Exception as err:
            return f"Error: {err}"

    @Slot(str, str, str)
    def saveReportFile(self, file_url, result_text, json_fields_text):
        try:
            user_input_data = json.loads(json_fields_text)
            with open(self._to_local_path(file_url), "w", encoding="utf-8") as f:
                f.write(
                    f"--- Rubber End AI Report ---\n{result_text}\nInput configuration used:\n"
                )
                for key, value in user_input_data.items():
                    f.write(f"  {key}: {value}\n")
        except Exception:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = RAIController()

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("backend", controller)
    engine.rootContext().setContextProperty("pysideApp", app)
    engine.addImportPath(os.path.dirname(__file__))

    qml_file = os.path.join(os.path.dirname(__file__), "main.qml")
    engine.load(QUrl.fromLocalFile(qml_file))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())
