import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: popup
    property string resultMessage: ""

    title: "Result"
    anchors.centerIn: parent

    contentWidth: 360
    standardButtons: Dialog.Save | Dialog.Close

    Connections {
        target: popup.footer
        function onChildrenChanged() {
            var saveBtn = popup.footer.standardButton(Dialog.Save)
            if (saveBtn) saveBtn.icon.name = "document-save"

                var closeBtn = popup.footer.standardButton(Dialog.Close)
                if (closeBtn) closeBtn.icon.name = "window-close"
        }
    }

    Label {
        text: popup.resultMessage
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.Wrap
        width: parent.width
        topPadding: 16
        bottomPadding: 16
        font {
            bold: true
            pointSize: 16
        }
    }

    onAccepted: {
        reportSaveFileDialog.open()
    }
}
