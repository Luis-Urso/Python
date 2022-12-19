# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:33:09 2022

@author: WB02554
"""

# echo-client.py

import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"Hello, world do Urso")
    data = s.recv(1024)

print(f"Received {data!r}")
