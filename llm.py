import re
import json
import requests
import asyncio
import aiohttp
import time

class Local_LLM():

    def __init__(self, ip_address, port, timeout_seconds=120, timeout_retry_increment=10):

        self.ip_address = ip_address
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.timeout_retry_increment = timeout_retry_increment

        self.model_parameters = asyncio.run(self.get_model_parameters())
        self.model_name = self.model_parameters['id']
        self.hosted_by = self.model_parameters['owned_by']

        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def get_model_parameters(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{self.ip_address}:{self.port}/v1/models") as response:
                response_data = await response.json()
                return response_data['data'][0]

    async def generate(self, message, temperature=0.0, break_pattern=None, **kwargs):
        
        if isinstance(message, str):
            message = [{'role': 'user', 'content': message}]
        
        url = f"http://{self.ip_address}:{self.port}/v1/chat/completions"

        # parameter differs between 'vllm' and 'sglang' models
        if self.hosted_by == "vllm":
            if "regex" in kwargs:
                kwargs["guided_regex"] = kwargs.pop("regex")
            if "max_new_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        elif self.hosted_by == "sglang":
            if "guided_regex" in kwargs:
                kwargs["regex"] = kwargs.pop("guided_regex")
            if "max_tokens" in kwargs:
                kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        data = {
            "messages": message,
            "temperature": temperature,
            "stop": break_pattern,
            "model": self.model_name,
            **kwargs
        }

        headers = {
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            response_text = None
            try:
                async with session.post(url, data=json.dumps(data), headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        response_text = response_json["choices"][0]['message']['content']
                        response_text = response_text.replace(message[0]['content'], "").strip()

                        # count tokens
                        prompt_tokens = response_json["usage"]["prompt_tokens"]
                        completion_tokens = response_json["usage"]["completion_tokens"]
                        self.prompt_tokens += prompt_tokens
                        self.completion_tokens += completion_tokens

                        # count latency
                        latency = int(time.time()) - response_json["created"]

                    else:
                        raise Exception(f"Failed to generate response: {response.text}")
            except asyncio.exceptions.TimeoutError:
                print(f"Request timed out after {self.timeout_seconds} seconds, retrying (+{self.timeout_retry_increment})...")
                self.timeout_seconds += self.timeout_retry_increment

        return {"response": response_text, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "latency": latency}
    
    def __call__(self, message, **kwds):
        return asyncio.run(self.generate(message, **kwds))
    
    def batch_inference(self, messages: list, batch_size=4, **kwargs):

        sem = asyncio.Semaphore(batch_size) # limit the number of concurrent requests

        async def sem_generate_response(message):
            async with sem:
                return await self.generate(message, **kwargs)

        async def gather_results(message_list):
            """gather results from all messages in message_list"""
            tasks = [sem_generate_response(m) for m in message_list]
            return await asyncio.gather(*tasks)

        # begin processing batch requests
        results = [None] * len(messages)    # initialize with None as all responses
        while None in results:
            unprocessed_messages = [m for m, r in zip(messages, results) if r is None]
            processed_results = asyncio.run(gather_results(unprocessed_messages))
            
            # place the processed results back into the original results list
            pr_idx = 0
            for idx in range(len(results)):
                if results[idx] is None:

                    if processed_results[pr_idx] is not None:
                        results[idx] = processed_results[pr_idx]

                    pr_idx += 1     # increment counter for processed_results whenever results[idx] is None
        return results

    def reset_token_count(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_token_count(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}
