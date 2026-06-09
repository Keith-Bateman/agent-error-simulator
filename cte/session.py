"""
CTESession — constructs per-agent Ollama base URLs when the CTE proxy is enabled.

When enabled, every agent routes its Ollama /api/chat calls through:
  http://{proxy_host}:{proxy_port}/_session/{session_id}
The proxy strips the /_session/{id} prefix, forwards to Ollama, and records
the full request/response as an InteractionRecord under that session ID.

When disabled, agents talk directly to Ollama.
"""

import logging

log = logging.getLogger(__name__)


class CTESession:
    def __init__(self, cfg):
        # type: (dict) -> None
        """
        cfg: the cte_proxy section from config.yaml
             { enabled: bool, host: str, port: int }
        """
        self.enabled = cfg.get("enabled", False)
        self.proxy_host = cfg.get("host", "localhost")
        self.proxy_port = int(cfg.get("port", 9090))

    def base_url(self, session_id, ollama_host, ollama_port):
        # type: (str, str, int) -> str
        """
        Return the base URL an agent should prepend to /api/chat.

        With proxy:  http://proxy_host:proxy_port/_session/{session_id}
        Without:     http://ollama_host:ollama_port
        """
        if self.enabled:
            url = "http://{}:{}/_session/{}".format(
                self.proxy_host, self.proxy_port, session_id
            )
            log.debug("CTE proxy routing: %s -> ollama:%d", url, ollama_port)
            return url
        return "http://{}:{}".format(ollama_host, ollama_port)

    def session_id(self, workflow_id, role, index=0):
        # type: (str, str, int) -> str
        """Canonical session ID format used across all agents."""
        return "{}_{}_{}" .format(workflow_id, role, index)
