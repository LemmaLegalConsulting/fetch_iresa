#!/bin/bash
# Quick setup script for AWS Lambda deployment

set -e

echo "🚀 AWS Lambda Deployment Setup"
echo "================================"

# Check prerequisites
echo "✓ Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it: https://aws.amazon.com/cli/"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install it: https://nodejs.org/"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install it."
    exit 1
fi

echo "✓ All prerequisites found"

# Ask for configuration
echo ""
echo "Configure AWS Lambda Deployment"
echo "==============================="

read -p "AWS Region [us-east-1]: " AWS_REGION
AWS_REGION=${AWS_REGION:-us-east-1}

read -p "Environment (dev/prod) [dev]: " ENVIRONMENT
ENVIRONMENT=${ENVIRONMENT:-dev}


read -p "Secret name in AWS Secrets Manager [legal-classifier/$ENVIRONMENT/secrets]: " SECRET_NAME
SECRET_NAME=${SECRET_NAME:-legal-classifier/$ENVIRONMENT/secrets}

# Import .env, filter out ENV, and convert to JSON
if [ -f .env ]; then
    echo "✓ Found .env file. Importing secrets (excluding ENV)..."
    TEMP_FILE=$(mktemp)
    echo '{' > "$TEMP_FILE"
    grep -v '^ENV=' .env | grep -v '^#' | grep -v '^$' | while IFS='=' read -r key value; do
        # Escape double quotes and backslashes in value
        esc_value=$(echo "$value" | sed 's/\\/\\\\/g; s/"/\\"/g')
        echo "  \"$key\": \"$esc_value\"," >> "$TEMP_FILE"
    done
    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "$TEMP_FILE"
    echo '}' >> "$TEMP_FILE"
    echo "✓ Converted .env to JSON:"
    cat "$TEMP_FILE"

    # Delete and re-import the secret
    echo ""
    echo "Re-importing secret $SECRET_NAME in AWS Secrets Manager..."
    if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" &>/dev/null; then
        echo "Secret exists. Deleting..."
        aws secretsmanager delete-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" --force-delete-without-recovery
        echo "Waiting for deletion..."
        sleep 8
    fi
    aws secretsmanager create-secret \
        --name "$SECRET_NAME" \
        --description "API keys for legal classifier $ENVIRONMENT environment" \
        --secret-string file://"$TEMP_FILE" \
        --region "$AWS_REGION"
    rm "$TEMP_FILE"
    echo "✓ Secret $SECRET_NAME imported from .env"
else
    echo "❌ .env file not found. Please create one with your secrets."
    exit 1
fi

# Install Node dependencies
echo ""
echo "Installing Serverless Framework and plugins..."
npm install -g serverless 2>/dev/null || echo "⚠️  Serverless already installed"
npm install --save-dev serverless-python-requirements 2>/dev/null || echo "✓ Plugins installed"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt -q
pip install -r requirements-dev.txt -q

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "==========="
echo ""
echo "1. Update your AWS credentials (if not already configured):"
echo "   aws configure"
echo ""
echo "2. Deploy to Lambda:"
echo "   STAGE=$ENVIRONMENT SECRETS_MANAGER_NAME=$SECRET_NAME serverless deploy --region $AWS_REGION"
echo ""
echo "3. View deployment info:"
echo "   serverless info --stage $ENVIRONMENT --region $AWS_REGION"
echo ""
echo "4. Stream logs:"
echo "   serverless logs -f api --stage $ENVIRONMENT --tail --region $AWS_REGION"
echo ""
echo "For more details, see LAMBDA_DEPLOYMENT.md"
