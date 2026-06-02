@echo off
REM build.bat — Build script for Windows
REM Requires: Visual Studio 2019+, CMake 3.15+

set BUILD_DIR=%~dp0build
set USE_OPENCL=OFF
set USE_CUDA=OFF

for %%a in (%*) do (
    if "%%a"=="--opencl" set USE_OPENCL=ON
    if "%%a"=="--cuda"   set USE_CUDA=ON
)

echo Flash-MoE Universal — Windows Build
echo =====================================
echo.

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DUSE_OPENCL=%USE_OPENCL% ^
    -DUSE_CUDA=%USE_CUDA% ^
    -DENABLE_SIMD=ON

cmake --build . --config Release

echo.
echo Build complete! Binary: %BUILD_DIR%\Release\flash_moe.exe
