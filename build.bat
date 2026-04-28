@echo off
echo Installing PyInstaller if needed...
py -m pip install pyinstaller

echo.
echo Building SEG_TEST.exe...
py -m PyInstaller --onefile --windowed --name SEG_TEST ^
    --hidden-import customtkinter ^
    --hidden-import CTkColorPicker ^
    SEG_TEST.py

echo.
echo Done! Output: dist\SEG_TEST.exe
pause
