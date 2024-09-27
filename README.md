# promptbreeder

This project implements the mechanism described in the [PromptBreeder](https://arxiv.org/abs/2309.16797) paper for automatic optimization and evaluation of prompts.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed git.
* You have Docker/Podman installed.

## Getting started

This project depends on another repository and uses Docker. Follow these steps to get it set up:

1. Clone this repository:
    ```bash
    git clone <repo>
    ```
2. Navigate into the project directory:
    ```bash
    cd promptbreeder/app/src
    ```
3. The project also has a prerequisite evaluation repository that needs to be cloned as a submodule into app/src:
    ```bash
    git submodule add <evaluation_module>
    ```
4. Run the Docker command:
    ```bash
    docker run -d --name <container_name> -v <path_to_your_project_on_host>/promptbreeder/app/src:/workspace/promptbreeder -e OPENAI_API_KEY=<your_openai_api_key> -it <docker_image_name>
    ```
5. Enter the Docker container:
    ```bash
    docker exec -it <container_name> /bin/bash
    ```
6. Navigate back to the main directory and install the project:
    ```bash
    cd /workspace/promptbreeder && pip install -e .
    ```
7. Execute the auto prompt optimization process
    ```bash
    python /workspace/promptbreeder/promptbreeder/auto_prompt_optimization.py
    ```
