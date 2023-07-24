from typing import Optional, List, Mapping, Any
import requests
from langchain.llms.base import LLM

import time

HOST = '192.168.94.20:8000'
URI = f'http://{HOST}/v1/chat/completions'


class LocalMPT(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["\n###", "\nObservation:"]

        response = requests.post(
            URI,
            json={
                "model": "gpt-3.5-turbo",
                "messages": prompt,
                "temperature": 0.7,
                "top_p": 1,
                "n": 1,
                "max_tokens": 64,
                "stop": stop,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "user": "string"
            },

        )
        response.raise_for_status()
        return response.json()['choices'][0]["message"]['content']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}


llm = LocalMPT()
start_time = time.time()
print(llm("What is Belarus?"))
print(time.time() - start_time)
