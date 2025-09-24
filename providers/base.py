from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Provider(ABC):
    name: str

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Devuelve solo el texto final de la respuesta.
        Espera una lista de mensajes con formato:
        {"role": "system|user|assistant", "content": "..."}
        """
        raise NotImplementedError
