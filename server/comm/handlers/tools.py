import time
from flask import jsonify
from server.comm.db import db_select, db_insert, db_delete


def verify_token(token):
    rows = db_select("token", "token", token,"TTL")
    if not rows:
        return False,jsonify({"type": "error","msg":"用户不存在"})
    rows = rows[0]
    username = rows["username"]
    db_delete("token","token",token)
    if rows["ttl"] < time.time():
        return False,jsonify({"type": "error","msg":"会话已过期"})
    db_insert("token",{"token":token,"TTL":time.time()+900,"username":username})
    return True, rows["username"]