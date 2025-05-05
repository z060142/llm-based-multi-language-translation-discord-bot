#!/bin/bash

# Discord Translation Bot Install Script for Linux/Ubuntu
# This script sets up the virtual environment and installs dependencies

echo "Discord Translation Bot Setup Script for Linux/Ubuntu"
echo "===================================================="

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    echo "For Ubuntu, run: sudo apt update && sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Check Python version
python_version=$(python3 -V 2>&1 | awk '{print $2}')
echo "Found Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install "discord.py>=2.3.2"
pip install "openai>=1.40.3"
pip install "pyyaml>=6.0.1"
pip install "aiosqlite>=0.19.0"
pip install "aiohttp>=3.9.5"

# Create start scripts
echo "Creating start scripts..."

# Create start script for bot only
cat > start-bot.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python bot.py
EOF
chmod +x start-bot.sh

# Create start script for UI
cat > start-ui.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python bot_ui.py
EOF
chmod +x start-ui.sh

echo ""
echo "Installation completed successfully!"
echo ""
echo "Available start scripts:"
echo "  ./start-bot.sh          - Start bot only"
echo "  ./start-ui.sh           - Start UI (can control bot from within UI)"
echo ""
echo "Next steps:"
echo "1. Configure your settings using the UI by running ./start-ui.sh"
echo "2. Use the UI to start/stop the bot as needed"
echo ""
echo "Note: If this is your first time, the bot will create a template config.yaml"
echo "      You'll need to fill in your Discord token and other API keys"
echo ""