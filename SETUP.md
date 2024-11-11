# RAG Docker Setup Procedure

## Initial Setup

Before building the project for the first time or after major changes:

1. Clean the environment: 
```bash
./cleanup.sh
```

2. Start the services:
```bash
./start_services.sh
```
This will:
- Verify clean environment
- Build and start Docker containers
- Initialize all services

## Common Issues

1. If you see permission or xattr errors:
   - Run `./cleanup.sh` first
   - Then run `./start_services.sh`

2. If you see symlink or metadata files (._*):
   - Run `./cleanup.sh` to remove them
   - Check that no symlinks exist before proceeding

## Development Workflow

1. First-time setup:
   ```bash
   git clone <repository>
   cd RAG-Docker
   ./cleanup.sh
   ./start_services.sh
   ```

2. After pulling new changes:
   ```bash
   ./cleanup.sh
   ./start_services.sh
   ```

3. When modifying Docker files:
   ```bash
   ./cleanup.sh
   ./start_services.sh
   ```

## Verification

You can verify the system is running correctly:
```bash
# Check container status
docker-compose -f docker/docker-compose.yml ps

# Check logs
docker-compose -f docker/docker-compose.yml logs
```