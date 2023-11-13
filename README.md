# upgraded-waffle
Complete Guide to TensorFlow with Deep Learning with Python. Udemy

Course by José PORTILLA.

https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python/learn/lecture/7798546?start=120#overview

Jupyter Notebooks written (or rewritten) as I go along.

Course materials (apparently all the general ones, given the directory name, 
'FULL_TENSORFLOW_NOTES_AND_DATA') available on my GitHub at

https://github.com/bballdave025/upgraded-waffle/tree/main/FULL_TENSORFLOW_NOTES__AND_DATA

Other useful stuff is on my Google Drive at

https://drive.google.com/drive/folders/1UVfSYw68tDpsLPdin1L6C5zopznnMPKb?usp=drive_link

```
thebballdave025 > My Drive > _Course_Reference > Quick_Dirs_Etc_Udemy_2023_Certs_TF
                > UdJoseCompleteTFForDLPython_Quick
                > UdJoseBroadTFDL_Section_001 > UdJoseBroadTFDL_001_Downloads
```

which has the directory, `FULL_TENSORFLOW_NOTES__AND_DATA`<strike>, and the file,
`tfdl_env.yml`</strike>.

<hr/>

### Note

For things to work in November 2023, I needed the workaround from Adam, on Udemy.

```
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>:: From Windows CMD
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>:: (Anaconda Prompt (miniconda3))
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>type tfdl_env.txt
# $ conda create --name tfdeeplearning --file tfdl_env.txt
# platform: win-64
@EXPLICIT
https://repo.anaconda.com/pkgs/main/win-64/_tflow_select-2.3.0-mkl.conda
https://repo.anaconda.com/pkgs/main/win-64/blas-1.0-mkl.conda
https://repo.anaconda.com/pkgs/main/win-64/ca-certificates-2020.1.1-0.conda
https://repo.anaconda.com/pkgs/main/win-64/icc_rt-2019.0.0-h0cc432a_1.conda
https://repo.anaconda.com/pkgs/main/win-64/intel-openmp-2019.4-245.conda
https://repo.anaconda.com/pkgs/msys2/win-64/msys2-conda-epoch-20160418-1.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/pandoc-2.2.3.2-0.conda
https://repo.anaconda.com/pkgs/main/win-64/vs2015_runtime-14.16.27012-hf0eaf9b_2.conda
https://repo.anaconda.com/pkgs/main/win-64/winpty-0.4.3-4.conda
https://repo.anaconda.com/pkgs/main/win-64/libmklml-2019.0.5-0.conda
https://repo.anaconda.com/pkgs/msys2/win-64/m2w64-gmp-6.1.0-2.tar.bz2
https://repo.anaconda.com/pkgs/msys2/win-64/m2w64-libwinpthread-git-5.0.0.4634.697f757-2.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/mkl-2018.0.3-1.conda
https://repo.anaconda.com/pkgs/main/win-64/vc-14.1-h0510ff6_4.conda
https://repo.anaconda.com/pkgs/main/win-64/icu-58.2-ha925a31_3.conda
https://repo.anaconda.com/pkgs/main/win-64/jpeg-9b-hb83a4c4_2.conda
https://repo.anaconda.com/pkgs/main/win-64/libsodium-1.0.16-h9d3ae62_0.conda
https://repo.anaconda.com/pkgs/msys2/win-64/m2w64-gcc-libs-core-5.3.0-7.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/openssl-1.0.2u-he774522_0.conda
https://repo.anaconda.com/pkgs/main/win-64/python-3.5.4-h1357f44_23.conda
https://repo.anaconda.com/pkgs/main/win-64/tbb-2020.0-h74a9793_0.conda
https://repo.anaconda.com/pkgs/main/win-64/zlib-1.2.11-h62dcd97_4.conda
https://repo.anaconda.com/pkgs/main/win-64/astor-0.7.1-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/backcall-0.1.0-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/certifi-2018.8.24-py35_1.conda
https://repo.anaconda.com/pkgs/main/noarch/colorama-0.4.3-py_0.conda
https://repo.anaconda.com/pkgs/main/noarch/decorator-4.4.2-py_0.conda
https://repo.anaconda.com/pkgs/main/noarch/defusedxml-0.6.0-py_0.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/entrypoints-0.2.3-py35_2.conda
https://repo.anaconda.com/pkgs/main/noarch/gast-0.3.3-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/ipython_genutils-0.2.0-py35ha709e79_0.conda
https://repo.anaconda.com/pkgs/main/win-64/libpng-1.6.37-h2a8f88b_0.conda
https://repo.anaconda.com/pkgs/main/win-64/libprotobuf-3.6.0-h1a1b453_0.conda
https://repo.anaconda.com/pkgs/msys2/win-64/m2w64-gcc-libgfortran-5.3.0-6.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/markdown-2.6.11-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/markupsafe-1.0-py35hfa6e2cd_1.conda
https://repo.anaconda.com/pkgs/main/win-64/mistune-0.8.3-py35hfa6e2cd_1.conda
https://repo.anaconda.com/pkgs/main/win-64/pandocfilters-1.4.2-py35_1.conda
https://repo.anaconda.com/pkgs/main/noarch/parso-0.7.0-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/pickleshare-0.7.4-py35h2f9f535_0.conda
https://repo.anaconda.com/pkgs/main/noarch/prometheus_client-0.7.1-py_0.tar.bz2
https://repo.anaconda.com/pkgs/main/noarch/pyparsing-2.4.7-py_0.conda
https://repo.anaconda.com/pkgs/main/noarch/pytz-2020.1-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/pywin32-223-py35hfa6e2cd_1.conda
https://repo.anaconda.com/pkgs/main/noarch/qtpy-1.9.0-py_0.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/send2trash-1.5.0-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/simplegeneric-0.8.1-py35_2.conda
https://repo.anaconda.com/pkgs/main/win-64/sip-4.18.1-py35h6538335_2.conda
https://repo.anaconda.com/pkgs/main/win-64/six-1.11.0-py35_1.conda
https://repo.anaconda.com/pkgs/main/win-64/sqlite-3.31.1-h2a8f88b_1.conda
https://repo.anaconda.com/pkgs/main/win-64/tbb4py-2018.0.5-py35he980bc4_0.conda
https://repo.anaconda.com/pkgs/main/win-64/termcolor-1.1.0-py35_1.conda
https://repo.anaconda.com/pkgs/main/noarch/testpath-0.4.4-py_0.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/tornado-5.1.1-py35hfa6e2cd_0.conda
https://repo.anaconda.com/pkgs/main/noarch/wcwidth-0.1.9-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/webencodings-0.5.1-py35_1.conda
https://repo.anaconda.com/pkgs/main/noarch/werkzeug-1.0.1-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/win_unicode_console-0.5-py35h56988b5_0.conda
https://repo.anaconda.com/pkgs/main/win-64/wincertstore-0.2-py35hfebbdb8_0.conda
https://repo.anaconda.com/pkgs/main/win-64/zeromq-4.2.5-he025d50_1.conda
https://repo.anaconda.com/pkgs/main/win-64/absl-py-0.4.1-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/cycler-0.10.0-py35hcc71164_0.conda
https://repo.anaconda.com/pkgs/main/win-64/freetype-2.8-h51f8f2c_1.conda
https://repo.anaconda.com/pkgs/main/win-64/jedi-0.12.1-py35_0.conda
https://repo.anaconda.com/pkgs/msys2/win-64/m2w64-gcc-libs-5.3.0-7.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/numpy-base-1.15.2-py35h8128ebf_0.conda
https://repo.anaconda.com/pkgs/main/win-64/protobuf-3.6.0-py35he025d50_0.conda
https://repo.anaconda.com/pkgs/main/noarch/python-dateutil-2.8.1-py_0.tar.bz2
https://repo.anaconda.com/pkgs/main/win-64/pyzmq-17.1.2-py35hfa6e2cd_0.conda
https://repo.anaconda.com/pkgs/main/win-64/qt-5.6.2-vc14h6f8c307_12.conda
https://repo.anaconda.com/pkgs/main/win-64/setuptools-40.2.0-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/traitlets-4.3.2-py35h09b975b_0.conda
https://repo.anaconda.com/pkgs/main/noarch/bleach-3.1.4-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/grpcio-1.12.1-py35h1a1b453_0.conda
https://repo.anaconda.com/pkgs/main/noarch/jinja2-2.11.2-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/jsonschema-2.6.0-py35h27d56d3_0.conda
https://repo.anaconda.com/pkgs/main/noarch/jupyter_core-4.5.0-py_0.conda
https://repo.anaconda.com/pkgs/main/noarch/pygments-2.6.1-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/pyqt-5.6.0-py35ha878b3d_6.conda
https://repo.anaconda.com/pkgs/main/win-64/pywinpty-0.5.4-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/wheel-0.31.1-py35_0.conda
https://repo.anaconda.com/pkgs/main/noarch/jupyter_client-5.3.3-py_0.conda
https://repo.anaconda.com/pkgs/main/noarch/nbformat-5.0.6-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/pip-10.0.1-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/prompt_toolkit-1.0.15-py35h89c7cb4_0.conda
https://repo.anaconda.com/pkgs/main/win-64/terminado-0.8.1-py35_1.conda
https://repo.anaconda.com/pkgs/main/win-64/ipython-6.5.0-py35_0.conda
https://repo.anaconda.com/pkgs/main/noarch/nbconvert-5.5.0-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/ipykernel-4.10.0-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/jupyter_console-5.2.0-py35_1.conda
https://repo.anaconda.com/pkgs/main/win-64/notebook-5.6.0-py35_0.conda
https://repo.anaconda.com/pkgs/main/noarch/qtconsole-4.7.4-py_0.conda
https://repo.anaconda.com/pkgs/main/win-64/widgetsnbextension-3.4.1-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/ipywidgets-7.4.1-py35_0.conda
https://repo.anaconda.com/pkgs/main/win-64/jupyter-1.0.0-py35_7.conda
https://repo.anaconda.com/pkgs/main/win-64/matplotlib-2.0.2-py35h9bd10b2_1.conda
https://repo.anaconda.com/pkgs/main/win-64/mkl_fft-1.0.6-py35hdbbee80_0.conda
https://repo.anaconda.com/pkgs/main/win-64/mkl_random-1.0.1-py35h77b88f5_1.conda
https://repo.anaconda.com/pkgs/main/win-64/numpy-1.15.2-py35ha559c80_0.conda
https://repo.anaconda.com/pkgs/main/win-64/pandas-0.20.3-py35he2ce742_2.conda
https://repo.anaconda.com/pkgs/main/win-64/scipy-1.1.0-py35h4f6bf74_1.conda
https://repo.anaconda.com/pkgs/main/win-64/tensorboard-1.10.0-py35he025d50_0.conda
https://repo.anaconda.com/pkgs/main/win-64/tensorflow-base-1.10.0-mkl_py35h81393da_0.conda
https://repo.anaconda.com/pkgs/main/win-64/scikit-learn-0.19.0-py35h3bd3ce1_2.conda
https://repo.anaconda.com/pkgs/main/win-64/tensorflow-1.10.0-mkl_py35h4a0f5c2_0.conda
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>
```

As it suggests in the comments at the top

```
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>:: To set up the environment
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>conda create --name tfdeeplearning --file tfdl_env.txt --yes
```

Then to activate it and start going with the Jupyter Notebooks, the next code snippets will do. (Note that I have notes available from the parent directory, so that's the place from which I launch the Jupyter server. I open up another Anaconda Prompt to the same directory - `upgraded-waffle` in this case - so that I can take care of my `git` stuff.)

```
(base) C:\bballdave025\my_repos_dwb\upgraded-waffle>conda activate tfdeeplearning
(tfdeeplearning) C:\bballdave025\my_repos_dwb\upgraded-waffle>cd ..
(tfdeeplearning) C:\bballdave025\my_repos_dwb>jupyter notebook
```

