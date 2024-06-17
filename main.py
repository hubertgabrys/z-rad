import sys
from multiprocessing import freeze_support

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QAction, QStyleFactory

from zrad.gui.filt_tab import FilteringTab
from zrad.gui.prep_tab import PreprocessingTab
from zrad.gui.rad_tab import RadiomicsTab
from zrad.gui.toolbox_gui import add_logo_to_tab, CustomWarningBox

WINDOW_TITLE = 'Z-Rad v8.0.dev'
WINDOW_WIDTH = 1800
WINDOW_HEIGHT = 750
BACKGROUND_COLOR = "#005ea8"
FONT_FAMILY = 'Verdana'
FONT_SIZE = 14


class ZRad(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tab_widget = None
        self.tabs = None
        self.menubar = None
        self.file_menu = None
        self.load_action = None
        self.save_action = None
        self.help_action = None
        self.exit_action = None

        self.init_gui()

    def init_gui(self):
        """
        Initialize the main GUI components.
        """
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.init_tabs()
        self.create_menu()

        self.show()

    def init_tabs(self):
        """
        Initialize and add tabs to the main window.
        """
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        self.tabs = [
            ("Resampling", PreprocessingTab()),
            ("Filtering", FilteringTab()),
            ("Radiomics", RadiomicsTab())
        ]

        for title, tab in self.tabs:
            tab.init_tab()
            self.tab_widget.addTab(tab, title)
            add_logo_to_tab(tab)

        self.tab_widget.currentChanged.connect(self.tab_changed)
        self.tab_widget.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")

    def create_menu(self):
        """
        Create and configure the menu bar.
        """
        self.menubar = self.menuBar()
        self.file_menu = self.menubar.addMenu('File')

        self.load_action = QAction('Load Input', self)
        self.save_action = QAction('Save Input', self)
        self.help_action = QAction('Help', self)
        self.exit_action = QAction('Exit', self)

        self.load_action.setShortcut('Ctrl+O')
        self.save_action.setShortcut('Ctrl+S')
        self.exit_action.triggered.connect(self.close)
        self.help_action.triggered.connect(self.display_help)

        self.file_menu.addAction(self.load_action)
        self.file_menu.addAction(self.save_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.help_action)
        self.file_menu.addAction(self.exit_action)

        self.load_action.triggered.connect(self.tabs[0][1].load_input_data)
        self.save_action.triggered.connect(self.tabs[0][1].save_input_data)

    @staticmethod
    def display_help():
        """
        Display help information.
        """
        text = (f'{WINDOW_TITLE} \n\nDeveloped at the University Hospital Zurich '
                'by the Department of Radiation Oncology\n\nAll relevant information about Z-Rad: ...')
        CustomWarningBox(text, False).response()

    def tab_changed(self, index: int):
        """
        Handle the change of tabs.
        """
        self.load_action.triggered.disconnect()
        self.save_action.triggered.disconnect()

        tab = self.tabs[index][1]
        self.load_action.triggered.connect(tab.load_input_data)
        self.save_action.triggered.connect(tab.save_input_data)

        self.load_action.setText('Load Input')
        self.save_action.setText('Save Input')


def main():
    if sys.platform.startswith('win'):
        freeze_support()

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(palette)

    app.setFont(QFont(FONT_FAMILY, FONT_SIZE))

    ex = ZRad()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
