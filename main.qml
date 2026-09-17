import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs
import "components"

ApplicationWindow {
    id: window
    visible: true
    width: 850
    height: 650
    title: "Rubber End AI"

    property string datasetPath: ""
    property string checkpointPath: ""
    property string inputPath: ""
    property int trainEpochs: 100
    property int trainPatience: 10
    property real trainLR: 0.0001

    property bool isCustomColors: backend ? backend.isCustomColors : false
    property bool isLightMode: backend ? backend.isLightMode : true

    Material.theme: isCustomColors ? (isLightMode ? Material.Light : Material.Dark) : Material.System
    Material.accent: isCustomColors ? "#fe8019" : window.palette.highlight
    Material.primary: isCustomColors ? (isLightMode ? "#f9f5d7" : "#3c3836") : window.palette.window
    Material.background: isCustomColors ? (isLightMode ? "#f9f5d7" : "#282828") : window.palette.window

    ListModel { id: predictParamsModel }

    StackView { id: stackView; anchors.fill: parent; initialItem: modeSelectionPage }

    Component { id: modeSelectionPage; MenuPage {} }
    Component { id: trainPage; TrainPage {} }
    Component { id: predictPage; PredictPage {} }

    TrainingParamsDialog { id: trainingParamsSettingsDialog }
    ResultPopup { id: resultDialog }

    readonly property url projectFolder: backend ? backend.getProjectPath() : ""

    FileDialog {
        id: datasetFileDialog;
        title: "RAI: Dataset file";
        currentFolder: backend ? backend.getProjectPath() : "";
        nameFilters: ["CSV Files (*.csv)"];
        onAccepted: {
            window.datasetPath = selectedFile.toString();
            checkpointSaveDialog.open()
        }
    }
    FileDialog {
        id: checkpointSaveDialog;
        title: "RAI: Checkpoint file";
        currentFolder: backend ? backend.getProjectPath() : "";
        fileMode: FileDialog.SaveFile; nameFilters: ["Model files (*.pth)"];
        onAccepted: {
            window.checkpointPath = selectedFile.toString();
            trainingParamsSettingsDialog.open()
        }
    }
    FileDialog {
        id: checkpointOpenDialog;
        title: "RAI: Checkpoint file";
        currentFolder: backend ? backend.getProjectPath() : "";
        nameFilters: ["Model files (*.pth)"];
        onAccepted: {
            window.checkpointPath = checkpointOpenDialog.selectedFile.toString();
            yamlOpenDialog.open()
        }
    }

    FileDialog {
        id: yamlOpenDialog; title: "RAI: Input file";
        currentFolder: backend ? backend.getProjectPath() : "";
        nameFilters: ["YAML files (*.yaml)"]
        onAccepted: {
            var fileContent = backend.loadYamlFile(selectedFile.toString());
            var parsedResult = backend.parseYamlToNative(fileContent);

            if (parsedResult === "INVALID_FORMAT" || parsedResult === "{}") {
                errorDialog.open();
            } else {
                window.inputPath = selectedFile.toString();
                stackView.push(predictPage);
            }
        }
    }

    FileDialog {
        id: reportSaveFileDialog; title: "RAI: Save Result";
        currentFolder: backend ? backend.getProjectPath() : "";
        fileMode: FileDialog.SaveFile;
        nameFilters: ["Text Files (*.txt)"]
        onAccepted: {
            var inputObj = {};
            for(var i = 0; i < predictParamsModel.count; ++i) {
                var item = predictParamsModel.get(i);
                inputObj[item.key] = item.value;
            }
            backend.saveReportFile(selectedFile.toString(), resultDialog.resultMessage, JSON.stringify(inputObj));
        }
    }


    Connections {
        target: backend
        function onTrainingFinished() {
            backend.saveCheckpoint(window.checkpointPath);
        }
    }
}
