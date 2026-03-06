import logging
import os
from datetime import datetime
logger = logging.getLogger(__name__)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_path = os.path.join(log_dir, log_file)

logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)