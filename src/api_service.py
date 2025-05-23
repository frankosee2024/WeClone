import os
import uvicorn
from llamafactory.chat import ChatModel
from llamafactory.api.app import create_app
from template import template_register
from utils.config import load_config

config = load_config('api_service')


template_register()


def main():
    chat_model = ChatModel(config)
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(os.environ.get("API_PORT", 8005)))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8005)), workers=1)


if __name__ == "__main__":
    main()
