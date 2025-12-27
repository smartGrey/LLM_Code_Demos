from flask import Flask, render_template, request
from cmd_interact import Bot
import sys
from pathlib import Path


# 因为这个脚本并非位于项目根目录下，所以无法直接饮用 config，需要先添加路径
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


# 启动项目
# 在终端执行：
# python ./app/app_server.py


app = Flask(__name__)
config = Config()
bot = Bot(config)


@app.route('/', strict_slashes=False)
def index():
    return render_template(config.html_template_name)


@app.route('/ask', methods=['POST'], strict_slashes=False)
def ask():
    user_input = request.form['user_input']
    response = bot.predict(user_input)
    return render_template(config.html_template_name, user_input=user_input, answer=response)


if __name__ == '__main__':
    app.run(debug=True)
