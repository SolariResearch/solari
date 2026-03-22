from setuptools import setup, find_packages

setup(
    name="solari-ai",
    version="0.1.0",
    description="The Deep Knowledge Engine — Turn anything into a searchable knowledge brain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Solari Systems",
    author_email="mark@solarisystems.net",
    url="https://github.com/SolariResearch/solari",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.0.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
    ],
    extras_require={
        "youtube": ["yt-dlp>=2023.0.0"],
        "pdf": ["pymupdf>=1.22.0"],
        "all": ["yt-dlp>=2023.0.0", "pymupdf>=1.22.0"],
    },
    entry_points={
        "console_scripts": [
            "solari=solari.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
