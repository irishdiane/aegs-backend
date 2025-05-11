#!/bin/bash
# This script sets up the environment, runs the Flask app, and exposes it using ngrok

set -e  # Exit on any error

echo "============================================"
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing ngrok..."
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc > /dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update
sudo apt install -y ngrok

echo "Adding ngrok authtoken..."
ngrok authtoken 2wuR54oxb54hVu2GOOTKj0I0g28_78Yxe5Vg7WzDb6SuFKy63

echo "Starting Flask application in background..."
flask run --host=0.0.0.0 --port=5000 &
FLASK_PID=$!

# Wait for Flask to start
sleep 3

echo "Starting ngrok tunnel..."
ngrok http 5000 > /dev/null &
NGROK_PID=$!

# Wait for ngrok to establish connection
sleep 3

NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')

echo ""
echo "========================================================"
echo "Flask app is running!"
echo "Your ngrok URL is: $NGROK_URL"
echo ""
echo "IMPORTANT: Update client/src/config.js in the aegs repo with:"
echo ""
echo "  const API_BASE_URL = \"$NGROK_URL\";"
echo ""
echo "Then commit and push to trigger a new GitHub Actions build."
echo "========================================================"

# Cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $FLASK_PID
    kill $NGROK_PID
    deactivate
    exit 0
}

trap cleanup INT

echo "Press Ctrl+C to stop services"
while true; do
    sleep 1
done
