#!/bin/bash
sudo apt-get update
sudo apt-get install -y pciutils
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
sudo systemctl status ollama
ollama pull llama3.2