import time
from flask import jsonify
from server.comm.db import db_select, db_insert


def verify_token(token):
    rows = db_select("token", "token", token,"TTL")
    if not rows:
        return False,jsonify({"type": "error","msg":"用户不存在"})
    row = rows[0]
    if row["TTL"] < time.time():
        return False,jsonify({"type": "error","msg":"会话已过期"})
    db_insert("token",{"token":token,"TTL":time.time()+900})
    return True, row["username"]