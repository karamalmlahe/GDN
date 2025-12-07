@echo off
echo Installing GDN dependencies...

echo.
echo Step 1: Installing core PyTorch packages...
pip install torch torchvision torchaudio

echo.
echo Step 2: Installing torch-geometric...
pip install torch-geometric

echo.
echo Step 3: Installing PyG extension libraries...
echo Note: This will install precompiled wheels for CPU. For GPU support, modify the URL accordingly.
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

echo.
echo Step 4: Installing additional dependencies...
pip install numpy scipy pandas scikit-learn matplotlib

echo.
echo Installation complete!
echo.
echo To run the project:
echo python main.py -dataset msl -save_path_pattern msl -slide_stride 1 -slide_win 5 -batch 32 -epoch 30 -comment msl -random_seed 5 -decay 0 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -decay 0 -val_ratio 0.2 -report best -topk 5 -device cpu
echo.
pause
