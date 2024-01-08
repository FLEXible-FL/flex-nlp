from setuptools import find_packages, setup


TF_requires = ["tensorflow<2.11", # https://github.com/tensorflow/tensorflow/issues/58973
                "tensorflow_datasets", 
                "tensorflow_hub"
        ]

HF_requires = ["datasets"]

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
                        "flexible",
                        "torch",
                        "torchtext",
                        "portalocker",
                        "torchdata",
                        ],
        extras_require={
                "tensorflow": TF_requires,
                "pytorch": PT_requires,
                "hugginface": HF_requires,
                "develop": ["pytest",
                        "pytest-cov",
                        "pytest-xdist",
                        "coverage",
                        "jinja2",
                        *TF_requires,
                        *HF_requires
                        ],
        },
        python_requires=">=3.8.10",
)
