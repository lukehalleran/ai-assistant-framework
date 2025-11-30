#!/bin/bash
# docker-compose-helper.sh
# Helper script for managing Docker Compose deployment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}WARNING: .env file not found${NC}"
        echo ""
        echo "Please create a .env file from the template:"
        echo "  cp .env.example .env"
        echo ""
        echo "Then edit .env and add your API keys:"
        echo "  OPENAI_API_KEY=your_key_here"
        echo ""
        read -p "Press Enter to continue anyway, or Ctrl+C to abort..."
    else
        echo -e "${GREEN}✓ .env file found${NC}"
    fi
}

# Build the image
build_image() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}Building Docker Image${NC}"
    echo -e "${BLUE}=====================================${NC}"
    docker-compose build
    echo -e "${GREEN}✓ Build complete${NC}"
}

# Start services
start_services() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}Starting Services${NC}"
    echo -e "${BLUE}=====================================${NC}"
    docker-compose up -d
    echo -e "${GREEN}✓ Services started${NC}"
    echo ""
    echo "Access the GUI at: http://localhost:7860"
    echo "Health check: http://localhost:7860/health"
    echo ""
    echo "View logs: docker-compose logs -f daemon-gui"
}

# Stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

# View logs
view_logs() {
    echo -e "${BLUE}Viewing logs (Ctrl+C to exit)...${NC}"
    docker-compose logs -f daemon-gui
}

# Check health
check_health() {
    echo -e "${BLUE}Checking health endpoint...${NC}"
    echo ""
    if curl -f http://localhost:7860/health 2>/dev/null; then
        echo ""
        echo -e "${GREEN}✓ Service is healthy${NC}"
    else
        echo ""
        echo -e "${RED}✗ Service is not responding or unhealthy${NC}"
        return 1
    fi
}

# Show status
show_status() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}Container Status${NC}"
    echo -e "${BLUE}=====================================${NC}"
    docker-compose ps
    echo ""
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}Volume Status${NC}"
    echo -e "${BLUE}=====================================${NC}"
    docker volume ls | grep daemon || echo "No daemon volumes found"
}

# Clean everything (including volumes)
clean_all() {
    echo -e "${RED}=====================================${NC}"
    echo -e "${RED}WARNING: This will delete ALL data${NC}"
    echo -e "${RED}=====================================${NC}"
    echo "This will remove:"
    echo "  - All containers"
    echo "  - All volumes (corpus, ChromaDB, logs)"
    echo "  - All cached data"
    echo ""
    read -p "Are you sure? Type 'yes' to confirm: " confirm
    if [ "$confirm" = "yes" ]; then
        echo -e "${YELLOW}Removing containers and volumes...${NC}"
        docker-compose down -v
        echo -e "${GREEN}✓ Cleanup complete${NC}"
    else
        echo -e "${YELLOW}Cleanup cancelled${NC}"
    fi
}

# Interactive CLI mode
run_cli() {
    echo -e "${BLUE}Starting interactive CLI mode...${NC}"
    echo "Type your messages and press Enter. Use Ctrl+D or 'exit' to quit."
    echo ""
    docker-compose run --rm daemon-gui cli
}

# Show help
show_help() {
    cat << EOF
${GREEN}=====================================${NC}
${GREEN}Daemon RAG Agent - Docker Helper${NC}
${GREEN}=====================================${NC}

Usage: $0 [command]

Commands:
  start       - Build and start services
  stop        - Stop services
  restart     - Restart services
  logs        - View service logs
  status      - Show container and volume status
  health      - Check health endpoint
  cli         - Run interactive CLI mode
  build       - Build Docker image only
  clean       - Stop and remove volumes (WARNING: deletes data)
  help        - Show this help message

Examples:
  $0 start      # First time setup
  $0 logs       # Monitor logs
  $0 health     # Check if running
  $0 cli        # Interactive chat

Environment:
  Create .env file from .env.example and add your API keys
  Required: OPENAI_API_KEY (or other LLM provider key)

For more info: https://github.com/yourusername/daemon-rag-agent
EOF
}

# Main command handler
case "${1:-help}" in
    start)
        check_env_file
        build_image
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        check_env_file
        start_services
        ;;
    logs)
        view_logs
        ;;
    status)
        show_status
        ;;
    health)
        check_health
        ;;
    cli)
        run_cli
        ;;
    build)
        build_image
        ;;
    clean)
        clean_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
