import os
import zipfile

import xbmc


def all(_in, _out, dp=None):
    try:
        root = os.path.realpath(_out)
        with zipfile.ZipFile(_in, 'r') as zin:
            items = zin.infolist()
            for count, item in enumerate(items, 1):
                target = os.path.realpath(os.path.join(root, item.filename))
                if target != root and not target.startswith(root + os.sep):
                    raise ValueError('Unsafe path in archive: %s' % item.filename)
                zin.extract(item, root)
                if dp:
                    dp.update(int(count * 100 / len(items)))
    except Exception as e:
        xbmc.log('[plugin.video.thokodi] %s' % e, xbmc.LOGERROR)
        return False
    return True
