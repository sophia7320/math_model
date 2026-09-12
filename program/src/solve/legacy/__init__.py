"""早期脚本（原样保留，勿改）。

这些脚本是 C 题最初的手写探索（2.py / solve1.py 等），依赖同目录下的
``data_reader`` / ``lp_model`` 以绝对模块名导入，运行方式：

    cd program
    uv run python src/solve/legacy/2.py

（脚本目录会被加入 sys.path，因此 data_reader / lp_model 可直接导入；
路径 ``./data/C/...`` 相对 cwd，必须在 program/ 下运行。）
"""
