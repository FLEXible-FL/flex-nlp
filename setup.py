from setuptools import find_packages, setup


PT_requires = ["torch", 
                "torchvision", 
                "torchtext", 
                "torchdata",
                "portalocker",
        ]

setup(
        name="flexnlp",
        version="0.0.1",
        author="Cristina Zuheros-Montes and Argente-Garrido Alberto",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="FL federated-learning flexible NLP",
        url="https://github.com/FLEXible-FL/flex-nlp",
        packages=find_packages(),
        install_requires=["numpy",
                        "multiprocess",
                        "scikit-learn",
                        "cardinality",
                        "sultan",
                        "tqdm",
                        "scipy",
                        "gdown",
                        # "flexible",
                        "torch",
                        "torchtext",
                        "portalocker",
                        "torchdata",
                        "datasets",
                        "transformers",
                        "sentence_transformers",
                        "sentencepiece",
                        ],
        extras_require={
                "pytorch": PT_requires,
                "develop": ["pytest",
                        "pytest-cov",
                        "pytest-xdist",
                        "coverage",
                        "jinja2",
                        ],
        },
        python_requires=">=3.8.10",
)
