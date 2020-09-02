import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cashocs",
    version="1.0.0",
    author="Sebastian Blauth",
    author_email="sebastian.blauth@itwm.fraunhofer.de",
    description="Computational Adjoint-Based Shape Optimization and Optimal Control Software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    # project_urls={'Source' : 'https://github.com/plugged/cashocs',
    #               'Documentation' : 'https://plugged.github.io/cashocs/docs/',
    #               'Tracker' : 'https://github.com/plugged/cashocs/issues'
    #               },
    packages=setuptools.find_packages(),
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords="Computational Adjoint-Based Shape Optimization and Optimal Control Software",
    install_requires=[
    	'meshio>=4.0.16,<=4.1.1',
		'pytest>=6.0.0'
    ],
    entry_points={
        "console_scripts" : [
            "cashocs-convert = cashocs._cli:convert"
        ]
    },
    python_requires='>=3.6',
)
