from utils import get_vlc2tok

PAD_INDEX = len(get_vlc2tok())
SOS_INDEX = len(get_vlc2tok()) + 1
EOS_INDEX = len(get_vlc2tok()) + 2
