from flask import Flask, request, jsonify

# 导入各个 handler 的对外入口函数
from handlers.upload_handler import handle_upload
from handlers.delete_handler import handle_delete
from handlers.login_handler import handle_login
from handlers.reg_handler import handle_reg
from handlers.question_handler import handle_question

app = Flask(__name__)

# type -> handler 的映射表
HANDLERS = {
    "upload": handle_upload,
    "delete": handle_delete,
    "login": handle_login,
    "register": handle_reg,
    "question":handle_question,
}

@app.route("/", methods=["POST"])
def handle_packet():
    data = request.get_json(force=True)

    if not data or "type" not in data:
        return jsonify({"error": "invalid packet"}), 400

    packet_type = data["type"]
    handler = HANDLERS.get(packet_type)

    if handler is None:
        return jsonify({"error": f"unknown type: {packet_type}"}), 400

    try:
        return handler(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765)
