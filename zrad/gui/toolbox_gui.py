from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ..toolbox_logic import tqdm_joblib


class CustomButton(QPushButton):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget, style: bool = True):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        if style:
            self.setStyleSheet("QPushButton {"
                               "background-color: #4CAF50; "
                               "color: white; "
                               "border: none; "
                               "border-radius: 25px;"
                               "}"
                               "QPushButton:hover {"
                               "background-color: green;"
                               "}"
                               )


class CustomBox(QComboBox):

    def __init__(self, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget, item_list: list):
        super().__init__(parent)
        for item in item_list:
            self.addItem(item)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("QComboBox:hover {"
                           "background-color: #27408B;"
                           "color: yellow; "
                           "}"
                           )


class CustomLabel(QLabel):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget,
                 style: str = "background-color: white; color: black; border: none; border-radius: 25px;"):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet(style)


class CustomInfo(QLabel):

    def __init__(self, text: str, info_tip: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("background-color: white; color: black; border: none; border-radius: 7px;")
        self.setToolTip(info_tip)


class CustomEditLine(QLineEdit):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("background-color: white; color: black;")


class CustomTextField(QLineEdit):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget, style: bool = False):
        super().__init__(parent)
        self.setPlaceholderText(text)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        if style:
            self.setStyleSheet("background-color: white; color: black; border: none; border-radius: 25px;")
        else:
            self.setStyleSheet("background-color: white; color: black;")


class CustomCheckBox(QCheckBox):

    def __init__(self, text: str, pos_x: int, pos_y: int, size_x: int, size_y: int, parent: QWidget):
        super().__init__(text, parent)
        self.setGeometry(pos_x, pos_y, size_x, size_y)
        self.setStyleSheet("""
            QCheckBox::indicator {
                width: 25px;
                height: 25px;
                border: 1px solid black;
                background-color: rgba(255, 255, 255, 128); /* Semi-transparent white background */
            }
            QCheckBox::indicator:checked {
                border: 1px solid black;
                background-color: rgba(144, 238, 144, 255); /* Semi-transparent white background */
            }
        """)


class CustomWarningBox(QMessageBox):

    def __init__(self, text: str, warning: bool = True):
        super().__init__()
        self.warning_key = warning
        self.setup_message_box(text)

    def setup_message_box(self, text: str):
        if self.warning_key:
            self.setIcon(QMessageBox.Warning)
            self.setWindowTitle('Warning!')
            self.setText(text)
            self.setStandardButtons(QMessageBox.Ok)
        else:
            self.setIcon(QMessageBox.Information)
            self.setWindowTitle('Help & Support')
            self.setText(text)
            self.setStandardButtons(QMessageBox.Close)

        self.setStyleSheet("QPushButton {"
                           "background-color: #FFD700;"
                           "color: black;"
                           "border-style: solid;"
                           "border-width: 2px;"
                           "border-radius: 10px;"
                           "border-color: #606060;"
                           "font: bold 16px;"
                           "padding: 10px;"
                           "}"
                           "QPushButton:hover {"
                           "background-color: #FF8C00;"
                           "}"
                           "QPushButton:pressed {"
                           "background-color: #505050;"
                           "}"
                           )
        font = QFont('Verdana')
        self.setFont(font)

    def response(self) -> bool:
        get_response = self.exec_()
        return get_response == QMessageBox.Ok


class CustomInfoBox(QMessageBox):

    def __init__(self, text: str, info: bool = True):
        super().__init__()
        self.info_key = info
        self.setup_message_box(text)

    def setup_message_box(self, text: str):
        if self.info_key:
            self.setIcon(QMessageBox.Warning)
            self.setWindowTitle('Info!')
            self.setText(text)
            self.setStandardButtons(QMessageBox.Ok)
        else:
            self.setIcon(QMessageBox.Information)
            self.setWindowTitle('Help & Support')
            self.setText(text)
            self.setStandardButtons(QMessageBox.Close)

        self.setStyleSheet("QPushButton {"
                           "background-color: #FFD700;"
                           "color: black;"
                           "border-style: solid;"
                           "border-width: 2px;"
                           "border-radius: 10px;"
                           "border-color: #606060;"
                           "font: bold 16px;"
                           "padding: 10px;"
                           "}"
                           "QPushButton:hover {"
                           "background-color: #FF8C00;"
                           "}"
                           "QPushButton:pressed {"
                           "background-color: #505050;"
                           "}"
                           )
        font = QFont('Verdana')
        self.setFont(font)

    def response(self) -> bool:
        get_response = self.exec_()
        return get_response == QMessageBox.Ok


class ProgressDialog(QDialog):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        self.progress_bar = QProgressBar(self)
        self.status_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def start(self, maximum: int, status_text: str = ""):
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(0)
        self.status_label.setText(status_text)
        self.show()
        QApplication.processEvents()

    def increment(self, step: int = 1, status_text: str | None = None):
        current_value = self.progress_bar.value()
        self.progress_bar.setValue(current_value + step)
        if status_text is not None:
            self.status_label.setText(status_text)
        # Ensure the UI updates during long-running tasks
        QApplication.processEvents()

    def finish(self, status_text: str = ""):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText(status_text)
        QApplication.processEvents()
        self.close()


class PatientProcessingWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, patient_folders, n_jobs, backend_hint, process_fn, progress_callback=None, parent=None):
        super().__init__(parent)
        self.patient_folders = patient_folders
        self.n_jobs = n_jobs
        self.backend_hint = backend_hint
        self.process_fn = process_fn
        self.progress_callback = progress_callback

    def run(self):
        try:
            results = []
            if self.n_jobs == 1:
                from tqdm import tqdm

                for patient_folder in tqdm(self.patient_folders, desc="Patient directories"):
                    status_text = f"Processing {patient_folder}"
                    if self.progress_callback:
                        self.progress_callback(1, status_text)
                    self.progress_updated.emit(1, status_text)
                    results.append(self.process_fn(patient_folder))
            else:
                from joblib import Parallel, delayed
                from tqdm import tqdm

                def _progress(step=1):
                    if self.progress_callback:
                        self.progress_callback(step, "Processing patients...")
                    self.progress_updated.emit(step, "Processing patients...")

                with tqdm_joblib(tqdm(desc="Patient directories", total=len(self.patient_folders)),
                                 progress_callback=_progress):
                    results = Parallel(n_jobs=self.n_jobs, prefer=self.backend_hint)(
                        delayed(self.process_fn)(patient_folder) for patient_folder in self.patient_folders)

            self.completed.emit(results)
        except Exception as exc:  # pragma: no cover - safety net for GUI thread
            self.failed.emit(str(exc))
