import datetime
import json
import mimetypes
import os
import re
import stat
from pathlib import Path

import humanize
from flask import make_response, request, render_template, send_file, Response
from flask.views import MethodView
from werkzeug.utils import secure_filename

from WebApp.FileServer import FLASK_APP, SERVER_ROOT, AUTH_COOKIE, IGNORED, DATA_TYPES, ICON_TYPES


@FLASK_APP.template_filter('size_fmt')
def size_fmt(size):
    return humanize.naturalsize(size)


@FLASK_APP.template_filter('time_fmt')
def time_desc(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


@FLASK_APP.template_filter('data_fmt')
def data_fmt(filename):
    t = 'unknown'
    for file_type, file_extents in DATA_TYPES.items():
        if filename.split('.')[-1] in file_extents:
            t = file_type
    return t


@FLASK_APP.template_filter('icon_fmt')
def icon_fmt(filename):
    i = 'fa-file-o'
    for icon, exts in ICON_TYPES.items():
        if filename.split('.')[-1] in exts:
            i = icon
    return i


@FLASK_APP.template_filter('humanize')
def time_humanize(timestamp):
    return humanize.naturaltime(datetime.datetime.utcfromtimestamp(timestamp))


def get_type(mode):
    if stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        path_type = 'dir'
    else:
        path_type = 'file'
    return path_type


def partial_response(path, start, end=None):
    file_size = os.path.getsize(path)

    if end is None:
        end = file_size - start - 1
    end = min(end, file_size - 1)
    length = end - start + 1

    with open(path, 'rb') as fd:
        fd.seek(start)
        response_bytes = fd.read(length)
    assert len(response_bytes) == length

    response = Response(
        response_bytes,
        206,
        mimetype=mimetypes.guess_type(path)[0],
        direct_passthrough=True,
    )
    response.headers.add(
        'Content-Range', 'bytes {0}-{1}/{2}'.format(
            start, end, file_size,
        ),
    )
    response.headers.add(
        'Accept-Ranges', 'bytes'
    )
    return response


def get_range(header_range):
    m = re.match('bytes=(?P<start>\d+)-(?P<end>\d+)?', header_range)
    if m:
        start = m.group('start')
        end = m.group('end')
        start = int(start)
        if end is not None:
            end = int(end)
        return start, end
    else:
        return 0, None


class PathView(MethodView):
    @classmethod
    def get(cls, p=''):
        hide_dotfile = request.args.get('hide-dotfile', request.cookies.get('hide-dotfile', 'no'))

        path = os.path.join(SERVER_ROOT, p)

        if os.path.isdir(path):
            contents = []
            total = {'size': 0, 'dir': 0, 'file': 0}
            for filename in os.listdir(path):
                if filename in IGNORED:
                    continue
                if hide_dotfile == 'yes' and filename[0] == '.':
                    continue
                filepath = os.path.join(path, filename)
                stat_res = os.stat(filepath)
                info = {'name': filename, 'mtime': stat_res.st_mtime}
                ft = get_type(stat_res.st_mode)
                info['type'] = ft
                total[ft] += 1
                sz = stat_res.st_size
                info['size'] = sz
                total['size'] += sz
                contents.append(info)
            page = render_template('index.html', path=p, contents=contents, total=total, hide_dotfile=hide_dotfile)
            res = make_response(page, 200)
            res.set_cookie('hide-dotfile', hide_dotfile, max_age=16070400)
        elif os.path.isfile(path):
            if 'Range' in request.headers:
                start, end = get_range(request.headers.get('Range'))
                res = partial_response(path, start, end)
            else:
                res = send_file(path)
                res.headers.add('Content-Disposition', 'attachment')
        else:
            res = make_response('Not found', 404)
        return res

    @classmethod
    def put(cls, p=''):
        if request.cookies.get('auth_cookie') == AUTH_COOKIE:
            path = os.path.join(SERVER_ROOT, p)
            dir_path = os.path.dirname(path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            info = {}
            if os.path.isdir(dir_path):
                try:
                    filename = secure_filename(os.path.basename(path))
                    with open(os.path.join(dir_path, filename), 'wb') as f:
                        f.write(request.stream.read())
                except Exception as e:
                    info['status'] = 'error'
                    info['msg'] = str(e)
                else:
                    info['status'] = 'success'
                    info['msg'] = 'File Saved'
            else:
                info['status'] = 'error'
                info['msg'] = 'Invalid Operation'
            res = make_response(json.JSONEncoder().encode(info), 201)
            res.headers.add('Content-type', 'application/json')
        else:
            info = {'status': 'error', 'msg': 'Authentication failed'}
            res = make_response(json.JSONEncoder().encode(info), 401)
            res.headers.add('Content-type', 'application/json')
        return res

    @classmethod
    def post(cls, p=''):
        if request.cookies.get('auth_cookie') == AUTH_COOKIE:
            path = os.path.join(SERVER_ROOT, p)
            Path(path).mkdir(parents=True, exist_ok=True)

            info = {}
            if os.path.isdir(path):
                files = request.files.getlist('files[]')
                for file in files:
                    try:
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(path, filename))
                    except Exception as e:
                        info['status'] = 'error'
                        info['msg'] = str(e)
                    else:
                        info['status'] = 'success'
                        info['msg'] = 'File Saved'
            else:
                info['status'] = 'error'
                info['msg'] = 'Invalid Operation'
            res = make_response(json.JSONEncoder().encode(info), 200)
            res.headers.add('Content-type', 'application/json')
        else:
            info = {'status': 'error', 'msg': 'Authentication failed'}
            res = make_response(json.JSONEncoder().encode(info), 401)
            res.headers.add('Content-type', 'application/json')
        return res

    @classmethod
    def delete(cls, p=''):
        if request.cookies.get('auth_cookie') == AUTH_COOKIE:
            path = os.path.join(SERVER_ROOT, p)
            dir_path = os.path.dirname(path)
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            info = {}
            if os.path.isdir(dir_path):
                try:
                    filename = secure_filename(os.path.basename(path))
                    os.remove(os.path.join(dir_path, filename))
                    os.rmdir(dir_path)
                except Exception as e:
                    info['status'] = 'error'
                    info['msg'] = str(e)
                else:
                    info['status'] = 'success'
                    info['msg'] = 'File Deleted'
            else:
                info['status'] = 'error'
                info['msg'] = 'Invalid Operation'
            res = make_response(json.JSONEncoder().encode(info), 204)
            res.headers.add('Content-type', 'application/json')
        else:
            info = {'status': 'error', 'msg': 'Authentication failed'}
            res = make_response(json.JSONEncoder().encode(info), 401)
            res.headers.add('Content-type', 'application/json')
        return res


def bind_url():
    path_view = PathView.as_view('path_view')
    FLASK_APP.add_url_rule('/', view_func=path_view)
    FLASK_APP.add_url_rule('/<path:p>', view_func=path_view)


if __name__ == '__main__':
    bind = 'localhost'
    port = '8050'
    FLASK_APP.run(bind, port, threaded=True, debug=False)
