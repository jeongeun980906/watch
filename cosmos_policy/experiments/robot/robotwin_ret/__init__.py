"""RoboTwin RAG inference (cross-embodiment retrieval-augmented).

Server-side counterpart of the cosmos_policy.experiments.robot.robotwin_zmq_server
EE module. The client-facing ZMQ protocol is identical — only difference is
that this server runs DINO + proprio retrieval against a pool of pool-embodiment
trajectories before invoking the policy, and injects the retrieved
(head, left_wrist, right_wrist, proprio, actions) into the obs that
cosmos_utils.get_action() consumes.
"""

from .retrieval import RoboTwinDinoProprioRetrieval

__all__ = ["RoboTwinDinoProprioRetrieval"]
