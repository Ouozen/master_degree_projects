import libtmux
import os
from tqdm.auto import tqdm
import argparse
from uuid import uuid4
import numpy as np


def start(name_session: str, num_users: int, base_dir='./'):
    """
    Запустить $num_users ноутбуков. У каждого рабочай директория $base_dir+$folder_num
    """
    # create tmux object
    server = libtmux.Server()
    session = server.new_session(name_session)
    log_str = ''
    ip = '0.0.0.0'

    # generate port
    flag = True
    while flag:
        port = np.random.randint(8000, 8888, num_users)
        if len(port) == len(set(port)): # all port number is unique
            flag = False

    for num_s in tqdm(range(num_users)):
        # create window and pane
        win = session.list_windows()
        win[-1].rename_window('window_' + str(num_s))
        pane = win[-1].list_panes()
        path = os.path.join(base_dir, 'folder_' + str(num_s))

        name_venv = 'venv_' + str(num_s)
        path_venv = os.path.join(path, name_venv)
        activate = os.path.join(path_venv, 'bin', 'activate')

        # generate token
        token = uuid4()

        # command
        pane[0].send_keys(f'mkdir {path}; python3 -m venv {path_venv}')
        pane[0].send_keys(f'source {activate}')
        pane[0].send_keys(f'jupyter notebook --ip {ip} --port {port[num_s]} --no-browser --NotebookApp.token={token} --NotebookApp.notebook_dir={path}') # run jupyter

        log_str += f'{num_s}: venv: {name_venv}, port: {port[num_s]}, token: {token}\n'

        if num_s < num_users - 1:
            session.new_window()
    else:
        print(log_str)
        print('All venv running.')

    

def stop(session_name: str, num: int):
    """
    @:param session_name: Названия tmux-сессии, в которой запущены окружения
    @:param num: номер окружения, кот. можно убить
    """
    server = libtmux.Server()
    session = server._list_sessions()
    for index, ses in enumerate(session):
        if ses['session_name'] == session_name:
            k = server.list_sessions()[index].list_windows()
            if len(k) == 1:
                server.kill_session(session_name)
                print('This is last window. Session killed.')
            else:
                server.list_sessions()[index].kill_window('window_' + str(num))
                print('Window stopped.')


def stop_all(session_name: str):
    """
    @:param session_name: Названия tmux-сессии, в которой запущены окружения
    """
    server = libtmux.Server()
    server.kill_session(session_name)
    print('kill session completed')


if __name__ == '__main__':
    # create arguments
    parser = argparse.ArgumentParser(description='command for script')
    parser.add_argument(
        'command',
        help='Name of command. Takes arguments like start, stop, stop_all.'
    )
    parser.add_argument(
        'num',
        type=int,
        nargs='?', default=0,
        help='Quantity or number to the command.'
    )
    arg = parser.parse_args()

    # run command
    name_session = 'session_tpos'
    if arg.command == 'start':
        start(name_session, arg.num)
    elif arg.command == 'stop':
        stop(name_session, arg.num)
    elif arg.command == 'stop_all':
        stop_all(name_session)
    else:
        print('Invalid command input.')