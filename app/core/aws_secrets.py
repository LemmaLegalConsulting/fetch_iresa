"""AWS Secrets Manager integration for Lambda environment variables.

This module handles loading API keys and secrets from AWS Secrets Manager
on Lambda cold starts, providing a centralized way to manage sensitive
configuration without using .env files in production.
"""

import json
import os
import logging
from functools import lru_cache
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_secrets_manager_client():
    """Get a cached boto3 Secrets Manager client."""
    return boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION", "us-east-1"))


def load_secret_from_aws(secret_name: str) -> Optional[Dict[str, str]]:
    """
    Load a secret from AWS Secrets Manager.

    Args:
        secret_name: The name or ARN of the secret in AWS Secrets Manager.

    Returns:
        A dictionary of key-value pairs, or None if the secret cannot be retrieved.
    """
    try:
        client = get_secrets_manager_client()
        response = client.get_secret_value(SecretId=secret_name)

        # The secret value can be either SecretString (JSON) or SecretBinary
        if "SecretString" in response:
            secret = response["SecretString"]
            try:
                # Try to parse as JSON
                return json.loads(secret)
            except json.JSONDecodeError:
                # If not JSON, return as a simple dict with 'value' key
                return {"value": secret}
        else:
            logger.warning(f"Secret {secret_name} is binary; skipping automatic load")
            return None

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ResourceNotFoundException":
            logger.warning(f"Secret {secret_name} not found in AWS Secrets Manager")
        elif error_code == "InvalidRequestException":
            logger.warning(f"Invalid request for secret {secret_name}")
        elif error_code == "InvalidParameterException":
            logger.warning(f"Invalid parameter for secret {secret_name}")
        else:
            logger.error(f"Error retrieving secret {secret_name}: {error_code}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading secret {secret_name}: {e}")
        return None


def load_lambda_secrets():
    """
    Load all required secrets from AWS Secrets Manager into environment variables.

    This function is meant to be called at Lambda cold start to populate
    environment variables from AWS Secrets Manager. It looks for a secret
    named by the SECRETS_MANAGER_NAME env var.

    Expected behavior:
    - If running on Lambda (check for LAMBDA_TASK_ROOT env var), load from Secrets Manager
    - If SECRETS_MANAGER_NAME is not set, skip loading and rely on env vars
    - Env vars already set take precedence over secrets manager values
    """
    # Only load from Secrets Manager if we're running on Lambda
    is_lambda = os.getenv("LAMBDA_TASK_ROOT") is not None

    if not is_lambda:
        logger.debug("Not running on Lambda; skipping Secrets Manager load")
        return

    secret_name = os.getenv("SECRETS_MANAGER_NAME")
    if not secret_name:
        logger.debug("SECRETS_MANAGER_NAME not set; skipping Secrets Manager load")
        return

    logger.info(f"Loading secrets from AWS Secrets Manager: {secret_name}")
    secrets = load_secret_from_aws(secret_name)

    if not secrets:
        logger.warning(f"Failed to load secrets from {secret_name}")
        return

    # Load secrets into environment, only if not already set
    env_keys = [
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "API_TOKENS",
    ]

    for key in env_keys:
        if key in secrets and key not in os.environ:
            os.environ[key] = secrets[key]
            logger.debug(f"Loaded {key} from Secrets Manager")

    logger.info("Secrets loaded successfully from AWS Secrets Manager")


# Call on module import (when Lambda initializes)
load_lambda_secrets()
