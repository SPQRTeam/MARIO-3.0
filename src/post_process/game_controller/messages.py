import base64
import struct

class SkippableGCMessage(Exception):
    pass

class GCReturnMessage_v4:
    THE_HEADER = b"RGrt"
    THE_VERSION = 4

    STRUCT_FORMAT = '< 4s b b b ? 3f f 2f'

    @classmethod
    def from_base64(cls, b64_string):
        msg_bytes = base64.b64decode(b64_string)
        msg_data = struct.unpack(cls.STRUCT_FORMAT, msg_bytes)
        if msg_data[0] != cls.THE_HEADER:
            raise ValueError("GC return message header doesn't match")
        if msg_data[1] != cls.THE_VERSION:
            # if received value looks like a legitimate version that should be implemented, use an exception that is not caught for a loud report
            if 1 <= msg_data[1] and msg_data[1] <= 10:
                raise ValueError(f"GC return message version doesn't match: expected {cls.THE_VERSION}, got {msg_data[1]}")
            # otherwise, it's probably garbage by some random misconfigured robot: use a special exception that will be caught and silently ignored
            else:
                raise SkippableGCMessage()
        return {
            "player_num": msg_data[2],
            "team_num": msg_data[3],
            "fallen": msg_data[4],
            "pose": {"x": msg_data[5], "y": msg_data[6], "theta": msg_data[7]},
            "ball_age": msg_data[8],
            "ball": {"x":msg_data[9], "y": msg_data[10]},
        }