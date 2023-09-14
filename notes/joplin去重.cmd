@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

set "folder_path=.\"  REM 将路径替换为实际的文件夹路径

for /r "%folder_path%" %%F in (*-1*) do (
    set "file=%%~nxF"
    set "new_name=!file:-1=!"
    if not exist "%%~dpF!new_name!" (
        ren "%%F" "!new_name!"
    ) else (
        del "%%F"
    )
)


echo 文件重命名完成！
pause