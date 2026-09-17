import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ColumnLayout {
        spacing: 8
        anchors {
            top: parent.top
            left: parent.left
            margins: 20
        }

        RowLayout {
            spacing: 8

            Label {
                text: "Colors: " + (window.isCustomColors ? "Gruvbox" : "System ")
                Layout.alignment: Qt.AlignVCenter
                Layout.fillWidth: true
            }

            Switch {
                checked: window.isCustomColors
                onToggled: backend.isCustomColors = checked
                Layout.alignment: Qt.AlignVCenter
            }
        }

        RowLayout {
            spacing: 8

            Label {
                text: "Mode: " + (window.isCustomColors
                ? (window.isLightMode ? "Light" : "Dark ")
                : "Auto ")
                Layout.alignment: Qt.AlignVCenter
                Layout.fillWidth: true
            }

            Switch {
                enabled: window.isCustomColors
                checked: !window.isLightMode
                onToggled: backend.isLightMode = !checked
                Layout.alignment: Qt.AlignVCenter
            }
        }
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 8
        Label {
            text: "Welcome to Rubber End AI."
            Layout.alignment: Qt.AlignHCenter
            font {
                pointSize: 24
                bold: true
            }
        }

        Label {
            text: "Select an operational mode to proceed:"
            Layout.alignment: Qt.AlignHCenter

        }

        RowLayout {
            spacing: 25;
            Layout.alignment: Qt.AlignHCenter

            Button {
                text: "Default Mode"
                icon.name: "media-playback-start"
                onClicked: checkpointOpenDialog.open()
            }

            Button {
                text: "Train Mode"
                icon.name: "applications-development"
                onClicked: datasetFileDialog.open()
            }
        }
    }
}
