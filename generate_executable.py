import os
import sys
import PyInstaller.__main__
import pydicom
import subprocess

# Detect platform
is_windows = sys.platform.startswith("win")
is_mac = sys.platform.startswith("darwin")

# Choose icon format
icon_path = "doc/logos/icon.icns" if is_mac else "doc/logos/icon.ico"

# Choose add-data separator
add_data_sep = ";" if is_windows else ":"

# Get the path to the pydicom data directory
pydicom_data_dir = os.path.join(os.path.dirname(pydicom.__file__), "data")

print(f"Pydicom data directory: {pydicom_data_dir}")

# Common PyInstaller arguments
common_args = [
    "main.py",
    f"--icon={icon_path}",
    f"--add-data=doc/logos{add_data_sep}doc/logos",
    f"--add-data={pydicom_data_dir}{add_data_sep}pydicom/data",
    "--hidden-import=pydicom.pixels.decoders.gdcm",
    "--hidden-import=pydicom.pixels.decoders.pylibjpeg",
    "--hidden-import=pydicom.pixels.decoders.pillow",
    "--hidden-import=pydicom.pixels.decoders.pyjpegls",
    "--hidden-import=pydicom.pixels.decoders.rle",
    "--hidden-import=pydicom.pixels.encoders.gdcm",
    "--hidden-import=pydicom.pixels.encoders.pylibjpeg",
    "--hidden-import=pydicom.pixels.encoders.native",
    "--hidden-import=pydicom.pixels.encoders.pyjpegls",
    "--log-level=DEBUG",
    "--clean",
]

# Platform-specific PyInstaller arguments
if is_windows:
    pyinstaller_args = [
        *common_args,
        "--onefile",
        "--name=Z-Rad",
    ]
elif is_mac:
    pyinstaller_args = [
        *common_args,
        "--windowed",  # no console window for GUI apps
        "--name=Z-Rad",
        "--osx-bundle-identifier=ch.usz.medphys.zrad",
    ]
else:
    raise RuntimeError("Unsupported platform for packaging.")

# Run PyInstaller
PyInstaller.__main__.run(pyinstaller_args)

# macOS: create DMG installer
if is_mac:
    app_path = os.path.join("dist", "Z-Rad.app")
    dmg_path = os.path.join("dist", "Z-Rad.dmg")
    if os.path.exists(app_path):
        print("Creating DMG installer...")
        subprocess.run([
            "hdiutil", "create",
            "-volname", "Z-Rad",
            "-srcfolder", app_path,
            "-ov",
            "-format", "UDZO",
            dmg_path
        ], check=True)
        print(f"DMG created: {dmg_path}")
    else:
        print("Z-Rad.app not found, skipping DMG creation.")
