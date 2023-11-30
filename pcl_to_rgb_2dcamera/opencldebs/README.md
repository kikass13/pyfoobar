

```
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-core_1.0.14828.8_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-opencl_1.0.14828.8_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu-dbgsym_1.3.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu_1.3.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd-dbgsym_23.30.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd_23.30.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/libigdgmm12_22.3.0_amd64.deb
```

sudo chmod -R 777 ./
sudo apt install ./*.deb

sudo gpasswd -a $USER render
sudo gpasswd -a $USER video

sudo reboot
