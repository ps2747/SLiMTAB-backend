#!/usr/bin/python3

import socket
import time
import struct
import threading

class SliMTABDriver:
    def __init__(self, ip=''):
        self.num_frets = 8
        self.status = 0
        self.s_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_cmd.bind(('', 5555))
        self.s_cmd.settimeout(1)
        self.dev_ip = ip
        self.conn = None

    # Return estimated transmiting time
    def check(self):
        start = time.perf_counter()
        if self._send_cmd(b'\xAB'):
            return (time.perf_counter() - start) / 2
        else:
            return -1

    def reset(self):
        return self._send_cmd(b'\xAB')

    def read(self):
        ts = struct.unpack("<I", self.conn.recv(4))[0]
        l = struct.unpack("<B", self.conn.recv(1))[0]
        d = self.conn.recv(l)
        ret = [0] * 6
        for dd in d:
            ret[dd//self.num_frets] = max(ret[dd//self.num_frets], dd%self.num_frets+1)
        return ts, l, ret
    def open(self):
        self.s_in.bind(('0.0.0.0', 5555))
        self.s_in.listen()
        self._send_cmd(b'\xAA')
        self.conn, _ = self.s_in.accept()

    def close(self):
        self._send_cmd(b'\xBB')
        self.conn.close()
        self.conn = None

    def begin(self):
        return self._send_cmd(b'\xCD')

    def end(self):
        return self._send_cmd(b'\xEF')

    def _send_cmd(self, cmd):
        try:
            self.s_cmd.sendto(cmd, (self.dev_ip, 6666))
            data, addr = self.s_cmd.recvfrom(1024)
            return True
        except ConnectionResetError:
            return False
        except socket.timeout:
            return False
        

############################
import signal
import sys
exit_loop = False
def signal_handler(signal, frame):
    global exit_loop
    exit_loop = True
        
signal.signal(signal.SIGINT, signal_handler)
############################

if __name__ == '__main__':
    instance = SliMTABDriver("192.168.100.1")
    delay = instance.check()
    if delay != -1:
        print('Latency: '+str(delay))
    else:
        sys.exit(1)

    instance.open()
    start = time.perf_counter()
    instance.reset()
    instance.begin()
    while not exit_loop:
        print(instance.read())
    print(time.perf_counter()-start)
    instance.end()
    instance.close()
    #instance.begin()

    #while not exit_loop:
    #    print(instance.read())

    #instance.end()



