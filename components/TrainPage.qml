import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: trainPage
    property int currentEpoch: 0
    property real progressValue: 0.0
    property string statusText: ""

    Connections {
        target: backend

        function onLogReceived(msg) {
            consoleText.append(msg)
            consoleText.cursorPosition = consoleText.length

            if (msg.indexOf("Epoch:") !== -1) {
                var match = msg.match(/Epoch:\s*(\d+)/)
                if (match && match[1]) {
                    var epochNum = parseInt(match[1])
                    trainPage.currentEpoch = epochNum
                    var totalEpochs = window.trainEpochs > 0 ? window.trainEpochs : 1
                    trainPage.progressValue = epochNum / totalEpochs
                    trainPage.statusText = "Training: Epoch " + epochNum + " / " + totalEpochs
                }
            }
            else if (msg.indexOf("[ SUCCESS ]") !== -1) {
                if (msg.indexOf("Saved") !== -1) {
                    trainPage.statusText = "Training finished successfully!"
                    trainPage.progressValue = 1.0
                } else if (msg.indexOf("Dataset loaded") !== -1) {
                    trainPage.statusText = "Dataset ready. Starting training..."
                }
            }
            else if (msg.indexOf("[ FAILED ]") !== -1 || msg.indexOf("[ ERROR ]") !== -1) {
                trainPage.statusText = "Training failed!"
            }
        }

        function onTrainingFinished() {
            if (trainPage.statusText.indexOf("failed") === -1 && trainPage.statusText.indexOf("Stopped") === -1) {
                trainPage.statusText = "Finished! Model saved."
                trainPage.progressValue = 1.0
            }
        }
    }

    ColumnLayout {
        spacing: 16
        anchors {
            fill: parent
            margins: 15
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 8
            visible: backend.isTraining || trainPage.progressValue > 0

            RowLayout {
                Layout.fillWidth: true
                Label {
                    text: trainPage.statusText
                    font.bold: true
                }

                Item {
                    Layout.fillWidth: true
                }

                Label {
                    text: Math.round(trainPage.progressValue * 100) + "%"
                    font.bold: true
                    color: (window && window.Material) ? window.Material.accent : "blue"
                }
            }

            ProgressBar {
                id: trainingProgressBar
                value: trainPage.progressValue
                Layout.fillWidth: true
                indeterminate: backend.isTraining && trainPage.currentEpoch === 0
            }
        }

        Label {
            text: "Console Output"
            font.bold: true
        }

        ScrollView {
            clip: true
            Layout.fillWidth: true
            Layout.fillHeight: true

            TextArea {
                id: consoleText
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.WrapAnywhere
                font.family: "Monospace"
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Button {
                text: "Main Menu"
                enabled: !backend.isTraining
                Layout.fillWidth: true
                onClicked: stackView.pop()
            }

            Button {
                text: "START"
                enabled: !backend.isTraining
                Layout.fillWidth: true
                onClicked: {
                    consoleText.clear()
                    trainPage.currentEpoch = 0
                    trainPage.progressValue = 0.0
                    trainPage.statusText = "Loading dataset..."

                    var datasetName = (typeof datasetPath === "string" && datasetPath) ? datasetPath.split('/').pop() : "Unknown";

                    consoleText.append("### Configuration Info ###\n" +
                        "Device: " + backend.getDevice() + "\n" +
                        "Dataset: " + datasetName + "\n" +
                        "Epochs: " + window.trainEpochs + "\n" +
                        "Patience: " + window.trainPatience + "\n" +
                        "Learning Rate: " + window.trainLR + "\n\n"
                    );

                    backend.startTraining(datasetPath, checkpointPath, window.trainEpochs, window.trainPatience, window.trainLR);
                }
            }

            Button {
                text: "STOP"
                enabled: backend.isTraining
                Layout.fillWidth: true
                onClicked: {
                    backend.stopTraining();
                    trainPage.statusText = "Stopped by user";
                    consoleText.append("\n[ WARN ]: Training stopped by user.");
                }
            }
        }
    }
}
