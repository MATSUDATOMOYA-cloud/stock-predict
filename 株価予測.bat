@echo off
chcp 65001 >nul
setlocal EnableExtensions
title 株価予測 (Random Forest)

rem ============================================================
rem  AIを使った株価の予想（ランダムフォレスト回帰）
rem  使い方 : このファイルをダブルクリック
rem           株価予測.bat 7203.T 2020-01-01 2025-07-17 のように引数も可
rem  同じフォルダの python\ にある Python で stock_predict.py を実行します。
rem  python\ が無い場合は、初回だけ自動でダウンロードして準備します。
rem  --update : yfinance を最新版に更新してから実行
rem  --reset  : python\ を削除して作り直す
rem ============================================================

set "PYVER=3.12.10"
set "PYTAG=python312"
set "RT=%~dp0python"
set "PY=%RT%\python.exe"
set "STOCK_OUT_DIR=%~dp0出力結果"

if /i "%~1"=="--reset" (
    echo 実行環境を削除して作り直します...
    if exist "%RT%" rmdir /s /q "%RT%"
)

if not exist "%RT%\ready.flag" (
    call :setup
    if errorlevel 1 goto :fail
)

if /i "%~1"=="--update" (
    echo yfinance を最新版に更新しています...
    "%PY%" -m pip install -U -q --no-warn-script-location yfinance
)

"%PY%" "%~dp0stock_predict.py" %*
set "ERR=%errorlevel%"
if defined STOCK_NO_PAUSE exit /b %ERR%
if "%ERR%"=="0" if exist "%STOCK_OUT_DIR%" start "" "%STOCK_OUT_DIR%"
echo.
pause
exit /b %ERR%

:setup
echo ============================================================
echo  初回セットアップ（数分かかります。インターネット接続が必要です）
echo  インストール先: %RT%
echo ============================================================
if exist "%RT%" rmdir /s /q "%RT%"
mkdir "%RT%" || exit /b 1
echo [1/3] Python %PYVER% をダウンロードしています...
curl.exe -# -fL -o "%RT%\python.zip" "https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip" || exit /b 1
tar -xf "%RT%\python.zip" -C "%RT%" || exit /b 1
del "%RT%\python.zip"
(echo %PYTAG%.zip& echo .& echo Lib\site-packages& echo import site)> "%RT%\%PYTAG%._pth"
echo [2/3] pip を準備しています...
curl.exe -# -fL -o "%RT%\get-pip.py" "https://bootstrap.pypa.io/get-pip.py" || exit /b 1
"%PY%" "%RT%\get-pip.py" -q --no-warn-script-location || exit /b 1
del "%RT%\get-pip.py"
echo [3/3] ライブラリをインストールしています...
"%PY%" -m pip install -q --no-warn-script-location -r "%~dp0requirements.txt" || exit /b 1
echo ok> "%RT%\ready.flag"
echo セットアップ完了
echo.
exit /b 0

:fail
echo.
echo セットアップに失敗しました。インターネット接続を確認して、もう一度実行してください。
pause
exit /b 1
