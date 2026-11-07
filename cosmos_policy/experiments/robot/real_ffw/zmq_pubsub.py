import zmq
import os
import time
import socket
import platform
import subprocess

def get_ip_address():
    """
    Get the local IP address of the machine.

    Returns:
        str: Local IP address as a string, or None if it cannot be determined.
    """
    system_name = platform.system()

    if system_name == "Darwin":  # macOS
        result = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    elif system_name == "Linux":  # Ubuntu or other Linux
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        ip_list = result.stdout.strip().split()
        return ip_list[0] if ip_list else None

    else:
        print(f"Unsupported system: {system_name}")
        return None
    
class ZMQPubSub(object):
    """
    A class for ZeroMQ publisher and subscriber sockets.

    Example
    -------
    Publisher side:
        pub = ZMQPubSub(role="pub",port=5556)
        ...
        pub.send({"qpos":qpos,"tick":tick})
        pub.close()

    Subscriber side:
        sub = ZMQPubSub(role="sub",port=5556)
        ...
        msg = sub.recv()
        if msg is not None:
            qpos = msg["qpos"]
            tick = msg["tick"]
        if sub.is_pub_closed():
            print("Publisher closed.")
        sub.close()
    """
    def __init__(
            self,
            role     = "pub",
            ip       = None,
            port     = 5556,
            bind_ip  = "0.0.0.0",
            hwm      = 1,
            conflate = True,
            verbose  = True,
        ):
        """
        Initialize a ZMQ publisher or subscriber socket.

        Parameters:
            role (str): "pub" for publisher, "sub" for subscriber.
            ip (str or None): IP address to connect to (subscriber) or bind to (publisher). If None, it will be auto-detected.
            port (int): Port number to connect/bind.
            bind_ip (str): IP address to bind for publisher (default "0.0.0.0").
            hwm (int): High water mark for the socket (default: 1).
            conflate (bool): Whether to enable message conflation (for subscribers). If True, only the most recent message is kept.
            verbose (bool): Whether to print verbose messages.
        """
        role = role.lower()
        assert role in ("pub", "sub"), "role must be 'pub' or 'sub'"
        self.context  = zmq.Context()
        self.socket   = None
        self.role     = role
        self.port     = port
        if ip is None: ip = get_ip_address()
        self.ip       = ip
        self.bind_ip  = bind_ip
        self.hwm      = hwm
        self.conflate = conflate
        self.verbose  = verbose
        self._pending_msg = None  # subscriber internal buffer
        self._pub_closed  = False # <-- 추가: publisher 종료 신호 수신 여부

        if self.role == "pub":
            self._open_pub()
        elif self.role == "sub":
            self._open_sub()
        else:
            raise ValueError("Invalid role: [%s]" % (self.role))

        if self.verbose:
            print ("[ZMQ] role:[%s] ip:[%s] port:[%d] hwm:[%d] conflate:[%s]"%
                   (self.role, self.ip, self.port, self.hwm, str(self.conflate)))

    def _open_pub(self):
        """
        Open a ZMQ publisher socket.
        """
        self.socket = self.context.socket(zmq.PUB)
        self.socket.set_hwm(self.hwm)
        self.socket.setsockopt(zmq.LINGER, 0) # prevent hanging on close
        endpoint = "tcp://%s:%d" % (self.bind_ip, self.port)
        self.socket.bind(endpoint)
        pid = os.getpid()
        print("[ZMQ Publisher] pid:[%d] endpoint:[%s]" % (pid, endpoint))

    def _open_sub(self):
        """
        Open a ZMQ subscriber socket.
        """
        self.socket = self.context.socket(zmq.SUB)
        self.socket.set_hwm(self.hwm)
        self.socket.setsockopt(zmq.LINGER, 0) # prevent hanging on close
        if self.conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)  # 가장 최근 메시지만 유지
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")   # 모든 토픽 구독
        endpoint = "tcp://%s:%d" % (self.ip, self.port)
        self.socket.connect(endpoint)
        pid = os.getpid()
        print("[ZMQ Subscriber] pid:[%d] endpoint:[%s]" % (pid, endpoint))

    def send(self, obj):
        """
        Send a Python object (publisher only).

        Parameters:
            obj: Any picklable Python object to send.
        """
        assert self.role == "pub", "Socket is not a publisher."
        self.socket.send_pyobj(obj)

    def recv(self, nonblock=True):
        """
        Receive a Python object (subscriber only).

        Parameters:
            nonblock (bool): If True, do not block if no message is available. Returns None if no message is received.
        """
        assert self.role == "sub", "Socket is not a subscriber."
        if self._pending_msg is not None:
            msg = self._pending_msg
            self._pending_msg = None
            return msg
        try:
            flags = zmq.NOBLOCK if nonblock else 0
            msg = self.socket.recv_pyobj(flags=flags)
            if isinstance(msg,dict) and msg.get("__ctrl__")=="EXIT":
                # EXIT message is not returned to user
                self._pub_closed = True   # <-- 추가: 종료 플래그 세팅
                self._pending_msg = None
                return None
            return msg
        except zmq.Again:
            return None

    def is_pub_closed(self, nonblock=True):
        """
        Check if the publisher has sent a close signal.

        Parameters:
            nonblock (bool): If True, do not block while checking for messages. Returns True if a close signal is detected, False otherwise.
        """
        assert self.role == "sub", "is_pub_closed() can only be used by subscriber."
        # 먼저 내부 플래그 확인
        if self._pub_closed:
            return True
        try:
            flags = zmq.NOBLOCK if nonblock else 0
            msg = self.socket.recv_pyobj(flags=flags)
            if isinstance(msg,dict) and msg.get("__ctrl__")=="EXIT":
                print("[ZMQ Subscriber] pid:[%d] Publisher closed detected."%(os.getpid()))
                self._pub_closed = True   # <-- 추가: 종료 플래그 세팅
                return True
            else:
                self._pending_msg = msg   # 일반 메시지는 버퍼에 저장
                return False
        except zmq.Again:
            return False

    def close(self):
        """
        Close the ZMQ socket and terminate the context.
        """
        if self.socket is not None:
            if self.role == "pub":
                try:
                    # send termination control message
                    self.socket.send_pyobj({"__ctrl__": "EXIT"})
                    print("[ZMQ Publisher] pid:[%d] sent close signal."%(os.getpid()))
                    time.sleep(0.01)  # <-- EXIT가 전달될 시간을 약간 확보
                except Exception as e:
                    print("[ZMQ Publisher] pid:[%d] failed to send close signal: %s"%(os.getpid(),str(e)))
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
        pid = os.getpid()
        print("[ZMQ %s] pid:[%d] Socket closed." % (self.role.upper(), pid))