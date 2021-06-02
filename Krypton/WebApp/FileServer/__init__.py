from flask import Flask

from ...Res.ToolKit import get_current_path

FLASK_APP = Flask('FileServer', static_url_path='/assets', static_folder='assets')
SERVER_ROOT = get_current_path().parent.parent.parent.joinpath('SavedBarData').name
AUTH_COOKIE = None
INDEX_FILE = get_current_path().parent.joinpath('templates', 'index.html')

IGNORED = ['.bzr', '$RECYCLE.BIN', '.DAV', '.DS_Store', '.git', '.hg', '.htaccess', '.htpasswd', '.Spotlight-V100', '.svn', '__MACOSX', 'ehthumbs.db', 'robots.txt', 'Thumbs.db', 'thumbs.tps']
DATA_TYPES = {'audio': 'm4a,mp3,oga,ogg,webma,wav', 'archive': '7z,zip,rar,gz,tar',
              'image': 'gif,ico,jpe,jpeg,jpg,png,svg,webp', 'pdf': 'pdf', 'quicktime': '3g2,3gp,3gp2,3gpp,mov,qt',
              'source': 'atom,bat,bash,c,cmd,coffee,css,hml,js,json,java,less,markdown,md,php,pl,py,rb,rss,sass,scpt,swift,scss,sh,xml,yml,plist',
              'text': 'txt',
              'video': 'mp4,m4v,ogv,webm', 'website': 'htm,html,mhtm,mhtml,xhtm,xhtml'}
ICON_TYPES = {'fa-music': 'm4a,mp3,oga,ogg,webma,wav',
              'fa-archive': '7z,zip,rar,gz,tar',
              'fa-picture-o': 'gif,ico,jpe,jpeg,jpg,png,svg,webp',
              'fa-file-text': 'pdf',
              'fa-code': 'atom,plist,bat,bash,c,cmd,coffee,css,hml,js,json,java,less,markdown,md,php,pl,py,rb,rss,sass,scpt,swift,scss,sh,xml,yml',
              'fa-file-text-o': 'txt',
              'fa-film': '3g2,3gp,3gp2,3gpp,mov,qt,mp4,m4v,ogv,webm',
              'fa-globe': 'htm,html,mhtm,mhtml,xhtm,xhtml'}


def bind_url():
    from .Server import bind_url
    bind_url()


bind_url()
