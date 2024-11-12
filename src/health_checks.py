import logging
import os

def setup_health_logger():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger('health_checks')
    # Prevent adding multiple handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('logs/health_checks.log')
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(file_handler)
        # Prevent propagation to root logger
        logger.propagate = False
    return logger

# Get logger instance
health_check_logger = setup_health_logger()

def check_service(service_name, ...):
    health_check_logger.info(f"Waiting for {service_name}...")
    # ... check logic ...
    health_check_logger.info(f"✓ {service_name} is ready (attempt {attempt})")