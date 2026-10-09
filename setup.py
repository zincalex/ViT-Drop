from setuptools import setup, find_packages


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def main():

    setup(
        name="vit-drop",
        version="1.0.0",
        author="Alessandro Viespoli, Loris Nanni",
        author_email="alessandro.viesp@gmail.com",
        description="Retraining-free depth pruning of vision transformers",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["vision transformer", "pruning", "depth pruning", "model compression", "ViT", "DINOv2", "SwinV2", "pytorch"],
        license="Apache 2.0 License",
        url="https://github.com/zincalex/ViT-Drop",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.10",
        install_requires=get_requires(),
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ]
    )


if __name__ == "__main__":
    main()
