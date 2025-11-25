from flask import jsonify

from server.comm.db import db_insert, db_select
from tools import verify_token
def handle_question(data):

    token = data["token"]
    check,username = verify_token(token)
    # 2. 如果返回的不是 True，说明失败了，直接把错误 Response 返回给前端
    if check is not True:
        return check

    timestamp = data["timestamp"]
    payload = data["payload"]
    db_insert("history",{"username":username,"timestamp":timestamp,"payload":payload,"role":"user"})
