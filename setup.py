"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import find_packages, setup

PT_requires = [
    "torch",
    "torchvision",
    "torchtext",
    "torchdata",
    "portalocker",
]

setup(
    name="flexnlp",
    version="0.1.0",
    author="Cristina Zuheros-Montes and Argente-Garrido Alberto",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="FL federated-learning flexible NLP",
    url="https://github.com/FLEXible-FL/flex-nlp",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "multiprocess",
        "scikit-learn",
        "cardinality",
        "sultan",
        "tqdm",
        "scipy",
        "gdown",
        "flexible-fl",
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
        "develop": [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "coverage",
            "jinja2",
        ],
    },
    python_requires=">=3.8.10",
)
