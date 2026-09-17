import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: dialog
    title: "Configuration"
    anchors.centerIn: parent
    standardButtons: Dialog.Ok | Dialog.Cancel

    GridLayout {
        columns: 2
        anchors.margins: 16
        rowSpacing: 8
        columnSpacing: 16

        Layout.fillWidth: true

        Label {
            text: "Epochs"
            font.bold: true
            Layout.alignment: Qt.AlignVCenter
        }

        SpinBox {
            id: epochsSpin
            from: 1
            to: 1000000
            value: 100
            editable: true
            Layout.fillWidth: true
        }

        Label {
            text: "Patience"
            font.bold: true
            Layout.alignment: Qt.AlignVCenter
        }

        SpinBox {
            id: patienceSpin
            from: 1
            to: 500
            value: 10
            editable: true
            Layout.fillWidth: true
        }

        Label {
            text: "Learning Rate"
            font.bold: true
            Layout.alignment: Qt.AlignVCenter
        }

        SpinBox {
            id: lrSpin
            from: 1
            to: 9999
            value: 1
            editable: true
            property real factor: 10000.0
            Layout.fillWidth: true

            textFromValue: function(value, locale) {
                return (value / factor).toLocaleString(locale, 'f', 4)
            }

            valueFromText: function(text, locale) {
                return Math.round(Number.fromLocaleString(locale, text.replace(",", ".")) * factor)
            }
        }
    }

    onAccepted: {
        var calculatedLR = lrSpin.value / lrSpin.factor;
        if (calculatedLR > 0) {
            window.trainEpochs = epochsSpin.value;
            window.trainPatience = patienceSpin.value;
            window.trainLR = calculatedLR;
            stackView.push(trainPage);
        }
    }
}
