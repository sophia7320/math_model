@echo off
chcp 65001 >nul
xelatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 pause & exit /b 1
xelatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 pause & exit /b 1
echo 编译完成：main.pdf
pause
