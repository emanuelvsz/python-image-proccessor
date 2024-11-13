import os
import boto3
import logging
import numpy as np
import json
import base64
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
BEDROCK_CLIENT = "bedrock-runtime"

CLAUDE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
CLAUDE_MODEL_ACCEPT = "application/json"
CLAUDE_MODEL_CONTENT_TYPE = "application/json"

AWS_BEDROCK_ACCESS_KEY=""
AWS_BEDROCK_SECRET_ACCESS_KEY=""
AWS_BEDROCK_REGION=""


class BedrockModel(ABC):
    @abstractmethod
    def execute(self, context, prompt=None):
        pass

    @abstractmethod
    def create_body(self, context, prompt=None):
        pass


class BedrockService:
    def __init__(self, aws_region: str, access_key_id: str, secret_access_key: str):
        self.__aws_region = aws_region
        self.__aws_access_key_id = access_key_id
        self.__aws_secret_access_key = secret_access_key
    
    def get_bedrock_client(self):
        try:
            session_kwargs = {
                "region_name": self.__aws_region,
                "aws_access_key_id": self.__aws_access_key_id,
                "aws_secret_access_key": self.__aws_secret_access_key,
            }
            return boto3.Session(**session_kwargs).client(BEDROCK_CLIENT)
        except Exception as e:
            logger.error("Failed to connect to Bedrock: %s", str(e))
            raise Exception("An error occurred while connecting the model")


class ClaudeHaikuLLMService(BedrockModel):
    def __init__(self):
        self.__bedrock_service = BedrockService(
            AWS_BEDROCK_REGION, AWS_BEDROCK_ACCESS_KEY, AWS_BEDROCK_SECRET_ACCESS_KEY
        )

    def execute(self, images, prompt):
        try:
            image_base64_list = [self.encode_image_to_base64(image) for image in images]
            body = self.create_body(image_base64_list, prompt)

            print("Sending request to Claude model with multiple images...")
            response = self.__bedrock_service.get_bedrock_client().invoke_model(
                modelId=CLAUDE_MODEL_ID,
                contentType=CLAUDE_MODEL_CONTENT_TYPE,
                accept=CLAUDE_MODEL_ACCEPT,
                body=body,
            )
            response_body = response.get("body").read()

            response_json = json.loads(response_body)

            content = response_json.get("content", [])
            if content:
                completion_text = content[0].get("text", "").strip()
                print("Response: ", completion_text)
                return completion_text
            else:
                return "No content in response"

        except Exception as e:
            raise RuntimeError("Failed to get response from Claude model") from e

    def encode_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded
        except Exception as e:
            raise

    def create_body(self, image_base64_list, prompt):
        content_items = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64
                }
            }
            for image_base64 in image_base64_list
        ]

        content_items.append({
            "type": "text",
            "text": prompt
        })

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": content_items
                    }
                ]
            }
        )
        return body

def process_all_images_in_directory(directory, prompt):
    service = ClaudeHaikuLLMService()
    images = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    service.execute(images, prompt)


if __name__ == "__main__":
    prompt = input("Digite o prompt para as imagens: ")
    process_all_images_in_directory('images', prompt)
