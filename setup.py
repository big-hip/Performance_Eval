from setuptools import setup, find_packages

setup(
    name="Performance_Eval",          # 包名（随你工程命名）
    version="0.1",                    # 版本号
    packages=find_packages(),         # 自动发现所有含 __init__.py 的包
    install_requires=[]             # 依赖库（按需写）
    
)
