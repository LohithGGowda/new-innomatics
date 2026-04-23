from setuptools import find_packages, setup

setup(
    name="rag_assistant",
    version="1.0.0",
    description="RAG-based Customer Support Assistant with LangGraph & HITL",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "rag-assistant=rag_assistant.main:main",
        ],
    },
)
