#!/usr/bin/env pythhon3

import sys
import logging
import os
import time
import socket
import requests
from string import Template
import json
import subprocess
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
from flask.helpers import make_response
import codes
import controller

app = Flask(__name__)
api = Api(app)

class ContainerHandler(Resource):

    def get(self):
        rc, msg = reqHandler.getContainersStat()
        return make_response(msg, codes.herror(rc))

    def post(self):
        reqData = request.get_json(force=True)
        rc,data = reqHandler.handleContainerOp(reqData)
        return make_response(data, codes.herror(rc))

class HostHandler(Resource):

    def get(self):
        reqData = request.get_json(force=True)
        rc,data = reqHandler.handleHostOp(reqData)
        return make_response(data, codes.herror(rc))

api.add_resource(ContainerHandler, '/container')
api.add_resource(HostHandler,'/host')

if __name__ == '__main__':
    cfg = dict()
    cfg["dockerurl"] = 'unix://var/run/docker.sock'
    port = 8081
    network = (subprocess.run("ip -o -4 addr show | awk '{print $1, $2, $4}'|grep 192.168.0", shell=True,stdout=subprocess.PIPE)).stdout.decode()
    ipaddr = ((network.split(' ')[2]).split('/')[0]).split('/')[0]
    cfg["interface"]  = network.split(' ')[1]
    cfg["hostIP"] = ipaddr
    reqHandler = controller.RequestRouter(cfg)
    try:
        app.run(
            host=ipaddr,
            port=int(port),
                    )
    except socket.error as msg:
        print(msg)