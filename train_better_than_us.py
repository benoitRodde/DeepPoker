import os
import sys
from pypokergui.server.poker import start_server
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket

sys.path.append("../../")




def main():
    tornado.options.parse_command_line()
    start_server("poker_conf.yaml", "8000", "fast")
    print(os.getcwd())


if __name__ == '__main__':
    main()
