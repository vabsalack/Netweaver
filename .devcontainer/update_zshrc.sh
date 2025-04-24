#!/bin/bash

# filepath: /home/USER_NETWEAVER/project_netweaver/add_zsh_plugins.sh

ZSHRC_PATH="$HOME/.zshrc"

# Ensure .zshrc exists
if [ ! -f "$ZSHRC_PATH" ]; then
    echo "Creating .zshrc file..."
    touch "$ZSHRC_PATH"
fi

# Add plugins to .zshrc
echo "Adding plugins to .zshrc..."
if ! grep -q "plugins=(" "$ZSHRC_PATH"; then
    echo "plugins=(git zsh-autosuggestions zsh-syntax-highlighting history)" >> "$ZSHRC_PATH"
else
    sed -i 's/plugins=(\(.*\))/plugins=(\1 zsh-autosuggestions zsh-syntax-highlighting history)/' "$ZSHRC_PATH"
fi

# Add theme to .zshrc
echo "Setting theme to powerlevel10k..."
if ! grep -q "ZSH_THEME=" "$ZSHRC_PATH"; then
    echo 'ZSH_THEME="powerlevel10k/powerlevel10k"' >> "$ZSHRC_PATH"
else
    sed -i 's/ZSH_THEME=.*/ZSH_THEME="powerlevel10k\/powerlevel10k"/' "$ZSHRC_PATH"
fi

echo "Plugins and theme added successfully!"
