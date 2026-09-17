import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: predictPage

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        Label {
            text: "Input Parameters"
            font {
                bold: true
                pointSize: 16
            }
        }

        ScrollView {
            id: paramsScrollView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            ListView {
                id: paramsList
                model: predictParamsModel
                spacing: 8
                width: paramsScrollView.availableWidth

                delegate: RowLayout {
                    width: paramsList.width
                    spacing: 16

                    Label {
                        text: model.key + ":"
                        font.bold: true
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                        Layout.alignment: Qt.AlignVCenter
                    }

                    TextField {
                        text: model.value
                        selectByMouse: true
                        Layout.preferredWidth: 260
                        Layout.alignment: Qt.AlignVCenter
                        onTextEdited: model.value = text
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            Button {
                text: "Main Menu"
                Layout.fillWidth: true
                Layout.preferredHeight: 45
                onClicked: stackView.pop()
            }

            Button {
                text: "Get a Prediction"
                Layout.fillWidth: true
                Layout.preferredHeight: 45

                onClicked: {
                    var inputObj = {}

                    for (var i = 0; i < predictParamsModel.count; ++i) {
                        var item = predictParamsModel.get(i)
                        var rawValue = item.value !== undefined && item.value !== null ? item.value : ""
                        var cleanStr = rawValue.toString().replace(",", ".").trim()

                        if (cleanStr.toLowerCase() === "true") {
                            inputObj[item.key] = true
                        } else if (cleanStr.toLowerCase() === "false") {
                            inputObj[item.key] = false
                        } else {
                            var numVal = Number(cleanStr)
                            inputObj[item.key] = (isNaN(numVal) || cleanStr === "") ? cleanStr : numVal
                        }
                    }

                    var result = backend.getPredictionFromFields(checkpointPath, JSON.stringify(inputObj))
                    resultDialog.resultMessage = result
                    resultDialog.open()
                }
            }
        }
    }

    Component.onCompleted: {
        var fileContent = backend.loadYamlFile(inputPath)
        var parsedResult = backend.parseYamlToNative(fileContent)

        predictParamsModel.clear()

        if (parsedResult !== "INVALID_FORMAT") {
            var parsedData = JSON.parse(parsedResult)
            for (var key in parsedData) {
                var initialValue = parsedData[key] !== null ? parsedData[key].toString() : ""
                predictParamsModel.append({ "key": key, "value": initialValue })
            }
        }
    }
}
