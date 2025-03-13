import os
import re
import subprocess
from pathlib import Path
import logging
from collections import defaultdict
import shutil

import yaml

logger = logging.getLogger(__name__)

# A flag to keep track if the Docker image has been checked
docker_image_checked = False


def check_and_pull_docker_image(docker_image):
    """
    Check if the Docker image is present, and pull it if it's not.
    """
    global docker_image_checked  # Use global to ensure the check is shared across calls

    if docker_image_checked:
        return

    try:
        subprocess.run(["docker", "inspect", "--type=image", docker_image],
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
        logger.info(f"Docker image {docker_image} is already present.")
    except subprocess.CalledProcessError:
        logger.info(
            f"Docker image {docker_image} not found. Pulling the image...")
        try:
            subprocess.run(["docker", "pull", docker_image], check=True)
            logger.info(f"Successfully pulled Docker image: {docker_image}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to pull Docker image {docker_image}. Error: {e}")
            raise

    # Set flag to True after the first successful check
    docker_image_checked = True


def parse_and_log_output(output: str, logger: logging.Logger, context: str):
    """
    Parse the output of a docker command and log messages at appropriate log levels.

    Args:
        output (str): The output from the subprocess.
        logger (logging.Logger): The logger to log the messages.
        context (str): Context for the logging (e.g., the image name or command being run).
    """
    if output:
        for line in output.splitlines():
            # Use regex to extract the log level and the actual message
            match = re.match(r'.*\[(\w+)\]:\s*(.+)', line)
            if match:
                log_level = match.group(
                    1).strip()  # Extract the log level part
                message = match.group(2).strip()  # Extract the message part
                if log_level == "CRITICAL":
                    logger.critical(f"{context}: {message}")
                elif log_level == "ERROR":
                    logger.error(f"{context}: {message}")
                elif log_level == "WARNING":
                    logger.warning(f"{context}: {message}")
                elif log_level == "INFO":
                    logger.info(f"{context}: {message}")
                elif log_level == "DEBUG":
                    logger.debug(f"{context}: {message}")
                else:
                    # Default to info if no log level is detected
                    logger.info(f"{context}: {message}")
            else:
                # Log the full line if it doesn't match the expected pattern
                logger.info(f"{context}: {line}")


def parse_histoqc_config_mapping(filepath, input_dir):
    """
    Parse the HistoQC config mapping YAML file and resolve wsi_ids to file paths in the input directory,
    filtering only valid WSI files (e.g., .svs, .tiff).

    Args:
        filepath (str): Path to the YAML file containing mappings.
        input_dir (Path): Path to the directory containing the input WSIs.

    Returns:
        list[dict]: List of dictionaries, each containing 'config' and 'wsi_paths'.
    """
    valid_extensions = {
        ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs",
        ".bif", ".svslide"
    }

    with open(filepath, 'r') as f:
        config_mapping = yaml.safe_load(f)

    mappings = config_mapping["mappings"]
    resolved_mappings = []

    for mapping in mappings:
        config_path = Path(mapping["config"])
        wsi_ids = mapping["wsi_ids"]

        # Search for matching WSI files in the input directory
        wsi_paths = []
        for wsi_id in wsi_ids:
            matching_files = [
                file for file in input_dir.rglob(f"*{wsi_id}*")
                if file.suffix.lower() in valid_extensions
            ]
            if not matching_files:
                logger.warning(
                    f"No valid WSI files found for WSI ID: {wsi_id}")
            else:
                wsi_paths.extend(matching_files)

        resolved_mappings.append({
            "config": config_path,
            "wsi_paths": wsi_paths,
        })

        file_list = []
        config_list = []
        for mapping in resolved_mappings:
            file_list.extend(mapping["wsi_paths"])
            config_list.extend([mapping["config"]] * len(mapping["wsi_paths"]))

        return file_list, config_list


def run_histoqc_task(input_dir, output_dir, num_workers, config,
                     search_pattern, wsi_id_mapping_style):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config = Path(config).resolve()

    if config.suffix == ".yaml" or config.suffix == ".yaml":
        file_list, config_list = parse_histoqc_config_mapping(
            config,
            input_dir,
        )
    file_list = Path(input_dir).rglob(search_pattern)


def run_histoqc(file_list,
                config_list,
                input_dir,
                output_dir,
                docker_image='histotools/histoqc:master',
                user=None,
                force=False,
                num_workers=None):
    """
    Process a given list of files with corresponding configuration files using HistoQC.
    
    Args:
        file_list (list[str]): List of files to process.
        config_list (list[str]): List of config files corresponding to the files.
        input_dir (Path): Path to the input directory containing the files.
        output_dir (Path): Path to the output directory.
        docker_image (str): Docker image for HistoQC.
        user (str): User identifier for the Docker container.
        force (bool): Force processing even if outputs already exist.
        num_workers (int): Number of workers for parallel processing.
    """
    if len(file_list) != len(config_list):
        raise ValueError(
            "The number of files must match the number of config files.")

    # Ensure Docker image is present
    check_and_pull_docker_image(docker_image)

    # Default user if not specified
    user = user or f'{os.getuid()}:{os.getgid()}'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    histoqc_output_dir = output_dir / "output"

    # Translate file paths to container paths
    container_input_dir = "/data_ro"
    container_file_list = [
        f"{container_input_dir}/{Path(file).relative_to(input_dir)}"
        for file in file_list
    ]

    # Group container file paths by configuration
    config_to_files = defaultdict(list)
    for container_file, config in zip(container_file_list, config_list):
        config_to_files[config].append(container_file)

    # List to store paths of renamed error logs
    error_logs = []

    # Process each config group
    for idx, (config_file,
              container_files) in enumerate(config_to_files.items()):
        histoqc_command = (f"histoqc_pipeline {' '.join(container_files)} "
                           f"-o /data/output -c {config_file}")
        if force:
            histoqc_command += " --force"

        if num_workers:
            histoqc_command += f" --n {num_workers}"

        command = [
            "docker", "run", "--rm", "-v",
            f"{input_dir}:{container_input_dir}:ro", "-v",
            f"{output_dir}:/data", "-v",
            f"{Path(config_file).parent}:{Path(config_file).parent}", "--name",
            f"histoqc_{Path(config_file).stem}_{idx}", "-u", user,
            docker_image, "/bin/bash", "-c", histoqc_command
        ]

        try:
            # Run Docker process
            result = subprocess.run(command, check=True)

            # Rename error.log to a unique name
            error_log_path = histoqc_output_dir / "error.log"
            if error_log_path.exists():
                unique_error_log_path = histoqc_output_dir / f"error_{idx}.log"
                shutil.move(error_log_path, unique_error_log_path)
                error_logs.append(unique_error_log_path)

            # Log results
            context = f"HistoQC output for config '{config_file}'"
            parse_and_log_output(result.stdout, logger, context)
            parse_and_log_output(result.stderr, logger, context)

            logger.info(f"Mask computed for files using config {config_file}.")

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error during mask computing for config {config_file}: {e}")
            raise

    # Concatenate all error logs into a single error.log
    if error_logs:
        final_error_log = histoqc_output_dir / "error.log"
        with open(final_error_log, "w") as outfile:
            for log in error_logs:
                with open(log, "r") as infile:
                    shutil.copyfileobj(infile, outfile)
        logger.info(f"Combined error log written to {final_error_log}.")

        # Delete the unique error log files
        for log in error_logs:
            log.unlink()
        logger.info("Unique error logs deleted after concatenation.")
