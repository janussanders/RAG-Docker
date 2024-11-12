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

def check_service(service_name, host, port, timeout=5, max_attempts=60):
    """
    Check if a service is ready by attempting to connect to its host and port.
    
    Args:
        service_name (str): Name of the service to check
        host (str): Host address of the service
        port (int): Port number of the service
        timeout (int): Timeout in seconds for each connection attempt
        max_attempts (int): Maximum number of connection attempts
    
    Returns:
        bool: True if service is ready, False otherwise
    """
    import time
    
    attempt = 1
    health_check_logger.info(f"Waiting for {service_name}...")
    
    # ... check logic ...
    health_check_logger.info(f"✓ {service_name} is ready (attempt {attempt})")