<p align="center">
    <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/branding/logo/primary/netweaverlogo-transparent.png" alt="netweaver logo" width=300 />
</p>

![GitHub License](https://img.shields.io/github/license/vabsalack/netweaver)



## Introduction

**Netweaver** is a lightweight, from-scratch implementation of neural networks, built primarily with [Numpy](https://numpy.org/) for efficient and fast matrix operations. It avoids heavy dependencies and complex abstractions, making it ideal for learning, experimentation, and educational use.

This project was born out of my own curiosity while exploring deep learning concepts. While popular frameworks offer powerful tools, they often come with layers of abstraction that can be difficult to modify or understand deeply. **Netweaver** aims to be different—modular, transparent, and flexible—providing a sandbox for implementing and testing your own ideas without the usual overhead.
Whether you're a student, a curious developer, or a researcher testing new ideas, Netweaver offers a flexible playground for experimentation and understanding the internals of neural networks.

The ultimate goal is to foster a collaborative space where learners and tinkerers can exchange ideas and build together. Contributions of all kinds—bug fixes, enhancements, or new features—are not only welcome but encouraged.

Source code: https://github.com/vabsalack/Netweaver  
discussions: https://github.com/vabsalack/Netweaver/discussions  
issues: https://github.com/vabsalack/Netweaver/issues  
community: https://discord.gg/5GS8EGdr  

To get started, simply [install](#installation) the library. For a hands-on introduction, I recommend downloading and running these two notebooks: [`demo_libusage.ipynb`](https://github.com/vabsalack/Netweaver/blob/main/notebooks/demonstration/demo_libusage.ipynb) and [`demo_metricmonitor.ipynb`](https://github.com/vabsalack/Netweaver/blob/main/notebooks/demonstration/demo_metricmonitor.ipynb). They'll walk you through the basics and show you how to make the most of Netweaver.


## Features

- No deep learning frameworks—just [Numpy](https://numpy.org/)
- Modular architecture (Layers, Activations, Losses, etc..)
- Easy to extend and customize
- Designed for learning and experimentation

## Table of Contents


| S.No. | Contents                                |
| ----- | --------------------------------------- |
|   1.    | [Introduction](#introduction)         |
|   2.    | [Features](#features)                 |
|   3.    | [Table of Contents](#table-of-contents)|
|   4.    | [Installation](#installation)         |
|   5.    | [Usage](#usage)                       |
|   6.    | [Project Structure](#project-structure)|
|   7.    | [Contributing](#contributing)         |
|   8.    | [Authors](#author)                    |
|   9.    | [Community](#community-channels)      |
|   10.   | [License](#license)                   |

## Installation

You can install Netweaver using:

### Option 1: [pip](https://pypi.org/project/pip/)
```bash
pip install netweaver
```
### Option 2: [uv](https://docs.astral.sh/uv/) (Recommended)
```bash
uv add netweaver
```
### Option 3: clone it locally for development and exploration.
```bash
git clone https://github.com/vabsalack/Netweaver.git
cd Netweaver
pip install -e .
```

## Usage
Here’s a minimal example of building and training a neural network with Netweaver:

For detailed instructions and examples, refer to the [`demo_libusage.ipynb`](https://github.com/vabsalack/Netweaver/blob/main/notebooks/demonstration/demo_libusage.ipynb) and [`demo_metricmonitor.ipynb`](https://github.com/vabsalack/Netweaver/blob/main/notebooks/demonstration/demo_metricmonitor.ipynb) files.

## Usage Overview & Highlights

Here are some key features that make working with Netweaver both intuitive and powerful:

1. **Effortless Dataset Handling:**  
    Netweaver offers built-in utilities to download and extract the Fashion MNIST dataset, allowing you to jump straight into building and training neural networks without manual data preparation. 
    <p align="center">
    <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/store_room/images/download.png" alt="download" />
    </p>

2. **Integrated Progress Monitoring:**  
    Training loops come with an embedded progress bar that displays real-time updates, including estimated time remaining for both epochs and batches. This provides clear visibility into training progress and helps manage expectations during long runs.
    <p align="center">
    <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/store_room/images/trainingstart.png" alt="progressbar1" />
    </p>
    <p align="center">
    <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/store_room/images/training_complete.png" alt="progressbar2" />
    </p>

3. **Real-Time Metric Visualization:**  
    The `netweaver.utils.PlotTraining` class provides real-time visualization of training and validation metrics. This feature is especially useful for tracking trends such as overfitting or underfitting, and for making timely decisions about early stopping. The plot automatically scrolls the x-axis to highlight the most recent metrics, ensuring clarity even as the graph crowds up. After training, you can display and save the complete graph of the training metrics to a .png file.

    <table>
        <tr>
            <td align="center">
                <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/store_room/images/graphstart.png" alt="Live metric animation" />
            </td>
            <td align="center">
                <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/store_room/images/graph_slide.png" alt="Scrolling metric graph" />
            </td>
            <td align="center">
                <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/store_room/images/statci_plot.png" alt="Static metric plot" />
            </td>
        </tr>
    </table>

4. **Organized Logging and Model Management:**  
    Each training session generates its own dedicated log folder. All logs and model files are automatically suffixed with a timestamp (`{now:%Y%m%d-%H%M%S}`), ensuring that results from different runs are kept separate and easy to compare. This organization simplifies experiment tracking and reproducibility.
    <p align="center">
    <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/store_room/images/log_dir.png" alt="directorylog" />
    </p>

These features collectively make Netweaver a practical and educational tool for anyone interested in understanding and experimenting with neural networks.

## Project Structure

```
.
|-- README.md
|-- pyproject.toml
|-- .gitignore
|-- .dockerignore
|-- branding
|-- notebooks
|   `-- demonstration
|       |-- demo_metricmonitor.ipynb
|       `-- demo_libusage.ipynb
|-- uv.lock
|-- src
|   `-- netweaver
|       |-- __init__.py
|       |-- accuracy.py
|       |-- datasets.py
|       |-- optimizers.py
|       |-- activation_layers.py
|       |-- model.py
|       |-- _internal_utils.py
|       |-- softmax_cce_loss.py
|       |-- utils.py
|       |-- layers.py
|       `-- lossfunctions.py
|-- .devcontainer
|   |-- devcontainer.json
|   |-- update_zshrc.sh
|   `-- Dockerfile
|-- LICENSE.txt
|-- .vscode
|   |-- extensions.json
|   `-- settings.json
|-- drawio
|-- store_room
|   |-- images
|-- tests
|   |-- e2e
|   |-- integration
|   |-- unit
|   |   |-- test_optimizers.py
|   |   |-- test_layers.py
|   |   |-- test_activation_layers.py
|   |   `-- test_loss_functions.py
|   `-- functional
|       |-- test_datasets.py
|       `-- test_model_training.py
`-- ruff.toml
```
## Contributing

We welcome all contributions! Whether you have ideas for improvements, bug fixes, or new features, your input is valued.

This project uses the Fork and Pull Request workflow:

1. **Fork** the repository to your own GitHub account.
2. **Create a new branch** for your changes (`git checkout -b feature/your-feature-name`).
3. **Make your changes** and commit them with clear, descriptive messages.
4. **Push** your branch to your forked repository.
5. **Open a Pull Request** to the main repository.

### 🐳 Development Environment: VS Code Dev Container

To streamline development and contributions, Netweaver provides a fully configured [VS Code Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) setup. This environment ensures consistency and reduces setup friction for all contributors.

#### Key Features

1. **Custom Dockerfile** ([Dockerfile](.devcontainer/Dockerfile))
    - Uses the fast and modern `uv` package manager for Python dependency management.
    - Installs a `uv`-managed Python interpreter (see Dockerfile for version details).
    - Configures the `zsh` shell with the [powerlevel10k](https://github.com/romkatv/powerlevel10k) theme for an enhanced terminal experience.

2. **Dev Container Configuration** ([devcontainer.json](.devcontainer/devcontainer.json))
    - Pre-installs essential VS Code extensions:
        - **Pylance** (Python language support)
        - **Jupyter** (notebook support)
        - **Ruff** (linting and formatting)
        - **Sourcery** (code quality suggestions)
        - **Draw.io Integration** (diagramming)
        - **Material Icon Theme** (file icons)
        - **Rainbow CSV** (CSV highlighting)

#### How It Works

The dev container uses a **bind mount** volume, meaning your project files and environment/cache data are stored on your local filesystem, outside the container. This ensures that dependencies, caches, and settings persist across container rebuilds and are not lost when the container is removed.

#### Troubleshooting

If you encounter any issues while setting up or using the dev container, please open a request or discussion in the [community](https://discord.gg/5GS8EGdr) channels or Github [discussions](https://github.com/vabsalack/Netweaver/discussions). 


Feel free to open issues for bugs, feature requests, or questions. Let's grow a collaborative community focused on learning and innovation!

### Contribution Guidelines

- As this is a new and evolving project, there are no strict contribution rules—just follow good Python practices.
- Please ensure your code is clean and readable before submitting a pull request.
- We use [Sourcery](https://sourcery.ai/) for code quality suggestions and [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Check the `ruff.toml` file for current rules or to suggest new ones.
- Be respectful and constructive in all discussions.

Thank you for helping make Netweaver better!

## Author

Netweaver was created as a side project to explore the deeper math behind neural networks and as a way to overcome boredom.
If you're curious about the inspiration behind Netweaver, check out the awesome [Neural Networks from Scratch](https://nnfs.io/) book. That's where the spark for this project came from

<table>
    <tr>
        <td width="90" align="center" valign="middle">
            <img src="https://avatars.githubusercontent.com/u/106925970?s=400&u=d865191ad6c0063ca9c3576ae3cca1df9707b1a3&v=4" alt="Author photo" width="70"  />
        </td>
        <td valign="middle">
            <b>keerthivasan</b>  
            <br>
            <a href="mailto:keerthi.pydev@gmail.com">Mail: Keerthi@gmail</a>  
            <br>
            <a href="https://github.com/vabsalack" target="_blank">GitHub: Vabsalack</a>
            <br>
            <a href="https://www.linkedin.com/in/keerthipydev">LinkedIn: Keerthivasan </a>
        </td>
    </tr>
</table>

### Contributors

<table>
    <tr>
        <td>
            <a href="https://github.com/vabsalack" target="_blank">Vabsalack</a>
        </td> 
    </tr>
</table>

## Community Channels

<table>
    <tr>
        <td>
            <a href="https://discord.gg/5GS8EGdr" target="_blank">Discord, click me</a>
        </td> 
        <td>
            use text channels wisely
        </td> 
    </tr>
</table>


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

<p align="center">
    <img src="https://raw.githubusercontent.com/vabsalack/Netweaver/refs/heads/main/branding/logo/logomark/netweaverlogoicon-transparent.png" alt="netweaver logo" width=70 />
</p>


