@echo off

REM Ensure the subfolder "static" exists
if not exist "static" mkdir "static"

REM Call the function to create and modify files
CALL :createAndModifyFile "pulse-modulation_pt1.ipynb" "static\pulse-modulation_pt1-static.ipynb"
CALL :createAndModifyFile "pulse-modulation_pt2.ipynb" "static\pulse-modulation_pt2-static.ipynb"
CALL :createAndModifyFile "pulse-modulation_pt3.ipynb" "static\pulse-modulation_pt3-static.ipynb"

REM Exit the script
exit /b

:createAndModifyFile

REM Copy the input file to the output file in the "static" folder
copy "%~1" "%~2"

REM Use PowerShell to modify the copied file's content
powershell -Command "(Get-Content -Raw -Path '%~2') -replace 'static_rendering = False', 'static_rendering = True' | Set-Content -Path '%~2%'"
powershell -Command "(Get-Content -Raw -Path '%~2') -replace 'title_text=', 'width=1200, title_text=' | Set-Content -Path '%~2%'"

echo The file "%~2" has been created with the changes applied.

GOTO :EOF
