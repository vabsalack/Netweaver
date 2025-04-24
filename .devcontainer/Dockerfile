ARG ARG_VARIANT="3.13.3-bookworm"
FROM python:${ARG_VARIANT}

# Define build arguments and environment variables
ARG ARG_UV_PYTHON="3.13.3"
ARG ARG_USER_NAME="USER_NETWEAVER"
ENV USER_NAME=${ARG_USER_NAME}
ENV HOME_DIR=/home/${USER_NAME}
ENV PATH="${HOME_DIR}/.local/bin:$PATH"

# Install zsh and create a new user
RUN apt-get update && apt-get install -y zsh && \
    useradd -m -s /bin/zsh ${USER_NAME} && \
    echo "${USER_NAME}:1234" | chpasswd && \
    chown -R ${USER_NAME}:${USER_NAME} ${HOME_DIR} && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER ${USER_NAME}

# Install UV and oh-my-zsh
WORKDIR ${HOME_DIR}
RUN curl -fsSL https://astral.sh/uv/install.sh -o uv-installer.sh && \
    curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -o oh-my-zsh-installer.sh && \
    sh uv-installer.sh && \
    sh oh-my-zsh-installer.sh --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ${HOME_DIR}/.oh-my-zsh/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${HOME_DIR}/.oh-my-zsh/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/romkatv/powerlevel10k.git ${HOME_DIR}/.oh-my-zsh/custom/themes/powerlevel10k 
    # && \ mkdir -p ${HOME_DIR}/.local/share/fonts && \
    # curl -fsSL https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/FiraMono/Regular/FiraMonoNerdFont-Regular.otf -o ${HOME_DIR}/.local/share/fonts/FiraMonoNerdFont-Regular.otf && \
    # curl -fsSL https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/FiraMono/Regular/FiraMonoNerdFontPropo-Regular.otf -o ${HOME_DIR}/.local/share/fonts/FiraMonoNerdFontPropo-Regular.otf



# Install Python version using UV
RUN uv python install ${ARG_UV_PYTHON}
WORKDIR ${HOME_DIR}/project_netweaver/


##################################### OLD/BACKUP FILE #####################################
# ARG ARG_VARIANT="3.13.3-bookworm"
# FROM python:${ARG_VARIANT}
# ARG ARG_UV_PYTHON="3.13.3" ARG_USER_NAME="USER_NETWEAVER"
# ENV USER_NAME=${ARG_USER_NAME}
# # add user & password and change ownership
# RUN useradd -m -s /bin/bash ${USER_NAME} && \
#     echo "${USER_NAME}:1234" | chpasswd && \
#     chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
# # Switch user
# USER ${USER_NAME}
# WORKDIR /home/${USER_NAME}/
# ADD --chown=${USER_NAME}:${USER_NAME} https://astral.sh/uv/install.sh uv-installer.sh
# RUN sh uv-installer.sh
# ENV PATH="/home/${USER_NAME}/.local/bin/:$PATH"
# RUN uv python install ${ARG_UV_PYTHON}
# WORKDIR /home/${USER_NAME}/project_netweaver/
