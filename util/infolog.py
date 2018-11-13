import atexit
import json
import os
from datetime import datetime
from threading import Thread
from urllib.request import Request, urlopen

from slackclient import SlackClient

_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None
_run_name = None
_slack_url = None


def init(filename, run_name, slack_url=None):
    global _file, _run_name, _slack_url, _slack_client
    _close_logfile()
    _file = open(filename, 'a', encoding="utf-8")
    _file.write('\n-----------------------------------------------------------------\n')
    _file.write('Starting new training run\n')
    _file.write('-----------------------------------------------------------------\n')
    _run_name = run_name
    _slack_url = slack_url
    slack_token = os.environ['SLACK_API_TOKEN']
    _slack_client = SlackClient(slack_token)


def log(msg, slack=True):
    print(msg)
    if _file is not None:
        _file.write('[%s]  %s\n' % (datetime.now().strftime(_format)[:-3], msg))
    if slack and os.environ.get('NOTIFY_SLACK', False) == 'True' and _slack_url is not None:
        Thread(target=_send_slack, args=(msg,)).start()


def upload_to_slack(audio_path, curr_step):
    Thread(target=_upload_wav_file, args=(audio_path, curr_step)).start()


def _close_logfile():
    global _file
    if _file is not None:
        _file.close()
        _file = None


def _upload_wav_file(audio_path, step):
    with open(audio_path, 'rb') as file_content:
        response = _slack_client.api_call(
            "files.upload",
            channels="multimodel_logs",
            file=file_content,
            title='Audio sample of a random image generated at step {}'.format(step)
        )

    if not response['ok']:
        _send_slack('Upload of audio sample Failed! Error: {}'.format(response['error']))


def _send_slack(msg):
    req = Request(_slack_url)
    req.add_header('Content-Type', 'application/json')
    urlopen(req, json.dumps({
        'username': 'Songbird-Net',
        'icon_emoji': ':taco:',
        'text': '*%s*: %s' % (_run_name, msg)
    }).encode())


atexit.register(_close_logfile)
