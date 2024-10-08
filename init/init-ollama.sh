#!/bin/bash
sudo apt-get update
sudo apt-get install -y pciutils
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
