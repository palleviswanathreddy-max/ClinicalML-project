@echo off
echo ===================================================
echo     ClinicalML Automated Deployment Script
echo ===================================================
echo.
set /p commit_msg="Enter your commit message (or press Enter to just use 'Update project'): "
if "%commit_msg%"=="" set commit_msg=Update project

echo.
echo Adding files to git...
git add .

echo.
echo Committing with message: "%commit_msg%"
git commit -m "%commit_msg%"

echo.
echo Pushing to GitHub (this will auto-trigger Vercel ^& GitHub Pages)...
git push origin main

echo.
echo ===================================================
echo DONE! Your websites will be updated in a minute.
echo ===================================================
pause




