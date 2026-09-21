import base64
import struct

class GCReturnMessage_v4:
    THE_HEADER = b"RGrt"
    THE_VERSION = 4

    STRUCT_FORMAT = '< 4s b b b ? 3f f 2f'

    @classmethod
    def from_base64(cls, b64_string):
        msg_bytes = base64.b64decode(b64_string)
        return cls.from_bytes(msg_bytes)

    @classmethod
    def from_bytes(cls, msg_bytes):
        msg_data = struct.unpack(cls.STRUCT_FORMAT, msg_bytes)
        if msg_data[0] != cls.THE_HEADER:
            raise ValueError("GC return message header doesn't match")
        if msg_data[1] != cls.THE_VERSION and msg_data[1] != -1:  # at least one occasion happened where a team set version -1. For now whatever, the version has been 4 ever since the new GC was made in 2023, so we can't have anything else at the moment.
            raise ValueError(f"GC return message version doesn't match: expected {cls.THE_VERSION}, got {msg_data[1]}")
        return {
            "player_num": msg_data[2],
            "team_num": msg_data[3],
            "fallen": msg_data[4],
            "pose": {"x": msg_data[5], "y": msg_data[6], "theta": msg_data[7]},
            "ball_age": msg_data[8],
            "ball": {"x":msg_data[9], "y": msg_data[10]},
        }
