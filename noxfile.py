import nox


@nox.session(venv_backend="mamba")
@nox.parametrize(
    "python,numpy,scipy,kwant",
    [
        ("3.10", "=1.23", "=1.9", "=1.4"),
        ("3.11", "=1.24", "=1.10", "=1.4"),
        ("3.12", ">=1.26", ">=1.13", ">=1.4"),
    ],
    ids=["minimal", "mid", "latest"],
)
def tests(session, numpy, scipy, kwant):
    session.run(
        "mamba",
        "install",
        "-y",
        f"numpy{numpy}",
        f"scipy{scipy}",
        f"kwant{kwant}",
        "packaging==22.0",
        "pytest-cov",
        "pytest-randomly",
        "pytest-repeat",
        "-c",
        "conda-forge",
    )
    session.install(".")
    session.run("pip", "install", "ruff", "pytest-ruff")
    session.run("pytest", "--ruff", "-x")
