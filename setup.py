from setuptools import setup, find_packages

setup(
    name="mobile-app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "pydantic-settings",
        "python-dotenv",
        "google-cloud-aiplatform",
        "motor",
        "fastapi",
        "websockets",
        "python-multipart",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "uvicorn"
    ],
) 