@echo off
title Push Flood Predictor to GitHub
color 0b
echo ===================================================
echo   AQUAGUARD AI - GITHUB UPLOADER
echo ===================================================
echo.
echo This script will help you push the project to your GitHub repository.
echo It will clear cached credentials of other accounts (like srisahitya15)
echo and prompt you to log in to your Shruthi-2244 account.
echo.
echo Press any key to continue...
pause > nul
echo.
echo [1/2] Clearing old cached credentials...
git credential-manager reject https://github.com
echo Credentials cleared.
echo.
echo [2/2] Pushing files to https://github.com/Shruthi-2244/flood_predictor.git...
git push -u origin main
echo.
echo ===================================================
echo   Upload process complete! Check your browser page.
echo ===================================================
echo.
echo Press any key to close this window.
pause > nul
