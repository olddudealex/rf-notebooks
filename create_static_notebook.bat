@echo off
REM Set the input file
set inputFile="pulse-modulation.ipynb"

REM Generate the name for the output file
set outputFile="pulse-modulation-static.ipynb"

REM Copy the input file to the output file
copy "%inputFile%" "%outputFile%"

REM Use PowerShell to modify the copied file's content
powershell -Command "(Get-Content -Raw -Path '%outputFile%') -replace 'static_rendering = False', 'static_rendering = True' | Set-Content -Path '%outputFile%'"

echo The file "%outputFile%" has been created with the changes applied.
